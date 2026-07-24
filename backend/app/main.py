"""FastAPI serving layer: upload an mp3, get the breakdown back.

Thin on purpose. All modelling lives in `model/` — this module only handles
HTTP concerns (multipart parsing, file hygiene, CORS) and assembles the
response. If you find yourself computing something here, it belongs in
`model/predictor.py` or `model/explain.py` instead.

    uvicorn backend.app.main:app --reload --port 8000
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Load the repo-root .env BEFORE importing anything that reads os.getenv at
# construction time (FameResolver picks up LASTFM_API_KEY in its __init__).
# Without this the backend saw no key at all and every unknown artist silently
# fell back to the low prior — a confident, plausible, wrong number.
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

from model.explain import audio_shap, generate_explanation, top_drivers  # noqa: E402
from model.fame import FameResolver  # noqa: E402
from model.features import AUDIO_FEATURES  # noqa: E402
from model.predictor import PredictorError, SongPredictor  # noqa: E402

log = logging.getLogger("soundsignal")

MAX_UPLOAD_MB = 25                      # mirrors the frontend's client-side limit
ALLOWED_SUFFIXES = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}

app = FastAPI(title="SoundSignal", version="1.0")

# Vercel's production origin gets added via env at deploy time; no code change.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        o.strip()
        for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
        if o.strip()
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module-level singletons: both load artifacts lazily on first use, and we want
# that cost paid once per process rather than once per request.
_predictor = SongPredictor()
_fame = FameResolver()

if not _fame.api_key:
    # Say this ONCE, loudly, at boot. Without a key the Last.fm step is skipped
    # entirely and every artist outside the local DB gets the p25 low prior — a
    # confident, plausible, wrong number. That used to surface only as a buried
    # per-request note, which is exactly how it went unnoticed.
    log.warning(
        "LASTFM_API_KEY is not set — artist lookups will SKIP Last.fm and fall "
        "back to the low fame prior (p25) for anyone missing from the local DB. "
        "Set it in %s to enable real fame resolution.",
        ROOT / ".env",
    )


@app.get("/health")
def health():
    """Liveness probe. The frontend polls this on page load to wake a sleeping
    host while the visitor is still reading, so it must stay cheap — no model
    loading here."""
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    audio_file: UploadFile = File(...),
    artist_name: str = Form(...),
    genre: str = Form(""),
    explicit: str = Form("false"),      # accepted for API compatibility; unused
):
    """Score one uploaded track and explain the result.

    `explicit` is accepted because the frontend sends it, but no current model
    consumes it — the context model is fame+genre and the audio model is
    librosa-only. Silently ignoring it beats pretending it matters.
    """
    suffix = Path(audio_file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            400,
            f"Unsupported file type {suffix or '(none)'}. "
            f"Expected one of: {', '.join(sorted(ALLOWED_SUFFIXES))}.",
        )
    if not artist_name.strip():
        raise HTTPException(400, "artist_name is required — fame is half the breakdown.")

    tmp_path = None
    try:
        # Stream to a temp file: librosa needs a real path, and we must not hold
        # a 25 MB upload in memory.
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(audio_file.file, tmp)
            tmp_path = tmp.name

        size_mb = os.path.getsize(tmp_path) / 1_048_576
        if size_mb > MAX_UPLOAD_MB:
            # Re-checked server-side: the client limit is a convenience, not a guard.
            raise HTTPException(
                413, f"File is {size_mb:.1f} MB — the limit is {MAX_UPLOAD_MB} MB."
            )

        fame = _fame.resolve(artist_name)
        context = {
            "artists_listeners": fame.listeners,
            # Empty string means "I don't know" -> predictor marginalizes over genres.
            "track_genre": genre.strip() or None,
        }

        try:
            prediction = _predictor.predict_from_audio_file(tmp_path, context=context)
        except PredictorError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            # Most often a decode failure on a corrupt/unsupported file.
            raise HTTPException(422, f"Could not analyze the audio: {exc}") from exc

        # Rebuild the exact scored rows for attribution. predict_from_audio_file
        # already ran librosa once and stashed the values, so this costs nothing.
        audio_features = {name: prediction.features[name] for name in AUDIO_FEATURES}
        audio_frame, _ = _predictor.frames_for(audio_features, context)

        try:
            shap_audio, _ = audio_shap(_predictor, audio_frame)
            shap_audio = top_drivers(shap_audio, n=6)
        except Exception:
            # An explanation is a nice-to-have; never fail a valid prediction for it.
            shap_audio = {}

        warnings = list(prediction.warnings)
        if fame.note:
            warnings.append(fame.note)

        # Pre-written caveat sentences, present ONLY when they actually apply.
        # The LLM used to receive `genre_imputed: false` and read the negated
        # field name as "the genre could not be imputed", then invented the rest.
        # A small model should never be handed a negative boolean to reason over:
        # an empty list has nothing to misread.
        caveats = []
        if fame.fame_estimated:
            caveats.append(
                "The artist was not found, so their fame is a low estimate (the "
                "25th percentile of known artists) and the score leans on the audio."
            )
        if prediction.genre_imputed:
            caveats.append(
                "No genre was given, so the style figure is averaged across all "
                "genres rather than being specific to this track."
            )

        # Pre-computed standing. The model read a raw 23.76 percentile as "above
        # average for hip-hop" — inverting the finding while sounding confident.
        # It is not asked to judge a number any more: it is handed the judgement,
        # with the number embedded, and the raw field is withheld from its view.
        pct = prediction.audio_percentile
        scope = prediction.audio_percentile_scope or "all genres"
        audio_standing = None
        if pct is not None:
            audio_standing = (
                f"above average — better than {round(pct)}% of {scope} tracks"
                if pct >= 50
                else f"below average — better than only {round(pct)}% of {scope} tracks"
            )

        # Same fix, generalized to fame/genre: the model separately inverted
        # genre_contribution +10.4 into "we need to subtract 10.4 points",
        # inventing a false premise ("hip-hop has a lower baseline than other
        # genres") that is nowhere in the payload. A signed float alone lets a
        # model override the sign with its own genre-popularity priors; a stated
        # fact leaves nothing to override.
        genre_name = context["track_genre"]
        genre_standing = None
        if genre_name and prediction.genre_contribution is not None:
            gc = prediction.genre_contribution
            genre_standing = (
                f"the {genre_name} genre performs ABOVE an average genre here, "
                f"adding {abs(round(gc, 1))} points"
                if gc >= 0
                else f"the {genre_name} genre performs BELOW an average genre here, "
                f"removing {abs(round(gc, 1))} points"
            )
        fc = prediction.fame_contribution
        fame_standing = (
            f"this artist's reach is ABOVE a typical artist's, adding {abs(round(fc, 1))} points"
            if fc >= 0
            else f"this artist's reach is BELOW a typical artist's, removing {abs(round(fc, 1))} points"
        )

        payload = {
            "predicted_popularity": prediction.predicted_popularity,
            "baseline": prediction.baseline,
            "fame_contribution": prediction.fame_contribution,
            "fame_standing": fame_standing,
            "genre_contribution": prediction.genre_contribution,
            "genre_standing": genre_standing,
            "audio_contribution": prediction.audio_contribution,
            "audio_standing": audio_standing,
            "audio_percentile": prediction.audio_percentile,
            "audio_percentile_scope": prediction.audio_percentile_scope,
            # The resolved genre by NAME, so the model never has to infer it.
            "genre": context["track_genre"],
            "caveats": caveats,
            # Derived from the waterfall split, NOT from a second SHAP pass. The
            # context model has only two inputs, so the split already is its full
            # attribution — and running TreeSHAP separately would produce a
            # slightly different (equally valid) division that visibly disagrees
            # with the tiles, especially when the genre is marginalized.
            "shap_context": {
                "fame": prediction.fame_contribution,
                "genre": prediction.genre_contribution,
            },
            "shap_audio": shap_audio,
            "warnings": warnings,
        }

        explanation, source = generate_explanation(payload)
        return {
            **payload,
            # Booleans stay in the RESPONSE (the frontend renders them as chips
            # and reads them correctly) — they are dropped only from the payload
            # the LLM sees, where a negated flag invites misreading.
            "fame_estimated": fame.fame_estimated,
            "genre_imputed": prediction.genre_imputed,
            "artist_fame": fame.listeners,
            "audio_features": audio_features,
            "explanation": explanation,
            "explanation_source": source,
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
