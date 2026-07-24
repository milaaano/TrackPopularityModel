r"""Serving-side entry point: features in, scores out.

This is the ONLY thing the backend should import. It owns the two artifacts
train.py produces and the contract between them:

    popularity  ≈  context_model(fame, genre)  +  audio_model(audio features)
                   \_______ what fame buys ____/    \___ what the SONG adds ___/

The audio model predicts the *residual* — the part of popularity that fame could
not explain. That is the song's own contribution, and it is deliberately small
(fame explains R²≈0.62, audio the leftovers). The SERVING audio model runs on
**librosa descriptors** extracted from the uploaded mp3 (Stage 6 Option B), via the
same model/audio.py extractor used in training, so there is no train/serve skew.
(The Spotify-feature version — residual Spearman ≈0.18 on 66k — is the research
track in notebooks/.) We report the audio part two ways (CLAUDE.md Stage 4):
  - raw points ("audio added +4"), which compose with context into the final
    popularity — intuitive, but +4 alone doesn't say whether +4 is good;
  - a PERCENTILE against all songs' audio contributions ("beats 73%"), which is
    what actually says whether +4 is good. The audio model was only ever graded
    on ranking (Spearman), so a rank output is the honest headline.

    from model.predictor import SongPredictor
    predictor = SongPredictor()          # artifacts load once, lazily
    result = predictor.predict_from_audio_file("song.mp3", context={...})
    result.to_dict()                     # JSON-ready for the API layer
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from model.features import (
    AUDIO_FEATURES,
    AUDIO_NUMERIC_FEATURES,
    AUDIO_CATEGORICAL_FEATURES,
    CONTEXT_FEATURES,
    FEATURE_DTYPES,
    FEATURE_RANGES,
)

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
CONTEXT_MODEL_FILE = "context_model.joblib"
AUDIO_MODEL_FILE = "audio_residual_model.joblib"
CALIBRATION_FILE = "audio_calibration.json"

POPULARITY_MIN, POPULARITY_MAX = 0, 100


class PredictorError(ValueError):
    """Bad input from the caller — safe to surface as a 400."""


@dataclass
class Prediction:
    # --- Audio model: this recording's own contribution ---
    # `audio_contribution` is the WITHIN-GENRE deviation ("craft"): the raw model
    # output minus that genre's mean output. The audio model's raw score is still
    # mildly genre-correlated (see calibrate.py), so subtracting the genre offset
    # keeps this number about the song rather than about its category.
    audio_percentile: float         # rank among songs OF THE SAME GENRE where possible
    audio_contribution: float       # popularity points, genre-centred ("craft added +2")
    audio_percentile_scope: str | None = None   # the genre it was ranked within, or "all genres"
    audio_contribution_raw: float | None = None  # before the genre offset (diagnostics)

    # --- Context: what fame + genre alone would buy this track ---
    context_contribution: float | None = None
    predicted_popularity: float | None = None   # context + audio, clipped to [0, 100]

    # --- The three-part breakdown (Stage 8) ---------------------------------
    # baseline + fame + genre + audio == predicted_popularity (before clipping).
    # Derived from the context model directly rather than SHAP: `genre` is how much
    # THIS genre differs from an average genre at this fame, and `fame` is how much
    # this fame differs from a reference-fame track. Both are exact, no attribution
    # library needed. None when no calibration artifact supplies the reference fame.
    baseline: float | None = None           # a median-fame track of an average genre
    fame_contribution: float | None = None  # what the artist's reach adds
    genre_contribution: float | None = None # what this style adds (incl. the audio genre offset)

    # Whether the context half of the answer can be trusted. False => report the
    # audio contribution alone; a popularity number without fame is not meaningful.
    context_available: bool = False
    # True when the caller didn't know the genre. The context prediction is then
    # MARGINALIZED — averaged over every genre the model was trained on — rather
    # than filled with one "typical" genre. Why not the mode: this dataset is
    # genre-balanced by construction, so the most-common genre is a near-tie won
    # by luck, and genres differ by tens of popularity points at fixed fame
    # (16–72 at 500k listeners). The average is the honest "I don't know".
    # Contrast model/fame.py, where "artist not found" IS informative (evidence
    # of obscurity) and the fill is deliberately LOW (p25). Stage 9 must phrase
    # the context part accordingly ("averaged across genres").
    genre_imputed: bool = False
    warnings: list[str] = field(default_factory=list)

    # Exactly the values that entered the models. Stage 9 (SHAP / LLM grounding)
    # must explain THESE numbers, not the caller's raw payload.
    features: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class SongPredictor:
    """Loads the trained artifacts once and scores songs from a feature payload."""

    def __init__(self, artifact_dir=None):
        self.artifact_dir = Path(artifact_dir) if artifact_dir else ARTIFACT_DIR
        self._context_model = None
        self._audio_model = None
        self._calibration = None
        self._known_genres = None
        self._mode_genre = None
        self._baseline = None   # waterfall base; constant, so computed once

    # ------------------------------------------------------------------ loading
    # Lazy + cached: a web worker imports this module at boot but should only pay
    # the unpickle cost on the first real request.
    def _load(self):
        if self._audio_model is not None:
            return
        import joblib

        audio_path = self.artifact_dir / AUDIO_MODEL_FILE
        context_path = self.artifact_dir / CONTEXT_MODEL_FILE
        for path in (audio_path, context_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing artifact {path}. Run: python -m model.train"
                )

        self._audio_model = joblib.load(audio_path)
        self._context_model = joblib.load(context_path)

        # The genres the context model's OneHotEncoder actually saw. An unseen
        # genre does not raise (handle_unknown="ignore") — it silently encodes to
        # an all-zero block, so we have to detect it ourselves to warn.
        categorical = self._context_model.named_steps["preprocessor"].named_transformers_[
            "categorical"
        ]
        self._known_genres = set(categorical.named_steps["encoder"].categories_[0])

        # The training-mode genre, learned by the categorical SimpleImputer when
        # the artifact was fit. NOT the unknown-genre fill (that path marginalizes
        # over all genres — see _marginal_context_prediction for why the mode
        # would be arbitrary here); kept as _frame's validation placeholder and
        # for UI display.
        self._mode_genre = str(categorical.named_steps["imputer"].statistics_[0])

        calibration_path = self.artifact_dir / CALIBRATION_FILE
        if calibration_path.exists():
            self._calibration = json.loads(calibration_path.read_text())
        else:
            self._calibration = None

    @property
    def known_genres(self):
        self._load()
        return sorted(self._known_genres)

    @property
    def mode_genre(self):
        """Most-common training genre (informational; unknown-genre predictions
        marginalize over all genres rather than filling this in)."""
        self._load()
        return self._mode_genre

    @property
    def context_model(self):
        """The fitted context pipeline. Public so model/explain.py can wrap it in
        a SHAP explainer without reaching into a private attribute."""
        self._load()
        return self._context_model

    @property
    def audio_model(self):
        """The fitted audio pipeline (same rationale as context_model)."""
        self._load()
        return self._audio_model

    # ------------------------------------------------------------- input hygiene
    def _frame(self, values, columns, kind):
        """One-row DataFrame with training dtypes. Raises on missing features.

        We do NOT let a missing feature fall through to the pipeline's imputer.
        The imputer exists for genuinely-missing values in the *training* data; at
        serving time a missing danceability means upstream extraction broke, and
        silently substituting the training median would return a confident score
        for a song we never actually looked at.
        """
        missing = [c for c in columns if values.get(c) is None]
        if missing:
            raise PredictorError(f"missing required {kind} features: {missing}")

        unexpected = set(values) - set(columns)
        row, warnings = {}, []
        for column in columns:
            value = values[column]
            low, high = FEATURE_RANGES.get(column, (-np.inf, np.inf))
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if value < low or value > high:
                    warnings.append(
                        f"{column}={value} outside training range [{low}, {high}]; clipped"
                    )
                    value = min(max(value, low), high)
            row[column] = value

        frame = pd.DataFrame([row], columns=columns)
        for column in columns:
            dtype = FEATURE_DTYPES.get(column)
            if dtype:
                try:
                    frame[column] = frame[column].astype(dtype)
                except (TypeError, ValueError) as exc:
                    raise PredictorError(f"{column!r} is not coercible to {dtype}: {exc}")
        if unexpected:
            warnings.append(f"ignored unknown {kind} keys: {sorted(unexpected)}")
        return frame, warnings

    # ------------------------------------------------------------- calibration
    def _to_percentile(self, residual, genre=None):
        """Rank a raw audio score, preferring the song's OWN genre as the field.

        Returns (percentile, scope). The calibration file stores 101 quantiles of
        the audio model's OUT-OF-FOLD predictions (see calibrate.py) — out-of-fold
        because in-sample predictions are over-spread, which would push every
        uploaded song toward the extremes of the scale.

        Why per genre: the same audio score means wildly different things by genre.
        Measured on ~9k songs, the globally-median score lands anywhere from the
        15th to the 78th percentile depending on the genre, so a global rank would
        tell a classical artist they are terrible and a pop artist they are fine
        for identical work. Genres with too few downloaded songs have no grid and
        fall back to the global scale, flagged.
        """
        if not self._calibration:
            return None, None
        per_genre = self._calibration.get("per_genre", {})
        if genre is not None and genre in per_genre:
            grid, scope = np.asarray(per_genre[genre], dtype=float), genre
        else:
            grid, scope = np.asarray(self._calibration["quantiles"], dtype=float), "all genres"
        # Fraction of that field this song's score exceeds.
        return float(np.searchsorted(grid, residual, side="right") / len(grid) * 100), scope

    def _genre_offset(self, genre):
        """The audio model's mean output for this genre (0.0 if unknown/ungridded).

        Subtracted from the raw score so `audio_contribution` is a within-genre
        deviation; the offset itself is added to the genre bar, so the total is
        unchanged and the waterfall identity still holds exactly.
        """
        if not self._calibration or genre is None:
            return 0.0
        return float(self._calibration.get("genre_offset", {}).get(genre, 0.0))

    def _context_split(self, context_frame, context_pred):
        """Split a context prediction into (baseline, fame part, genre part).

        Exactly additive by construction:
            baseline          = marginal context at a REFERENCE fame  (average genre)
            fame              = marginal context at THIS fame - baseline
            genre             = this genre's context - marginal at this fame
            baseline + fame + genre == context_pred

        "Marginal" means averaged over every training genre, so each term isolates
        one variable. When the genre is unknown we marginalize anyway, and the genre
        term falls out to ~0 on its own — no special case needed.
        """
        reference_fame = (self._calibration or {}).get("reference_fame")
        if reference_fame is None:
            return None, None, None

        marginal_here = self._marginal_context_prediction(context_frame)
        if self._baseline is None:
            reference_frame = context_frame.copy()
            reference_frame["artists_listeners"] = float(reference_fame)
            # Constant across requests, so compute once.
            self._baseline = self._marginal_context_prediction(reference_frame)
        return self._baseline, marginal_here - self._baseline, context_pred - marginal_here

    def _marginal_context_prediction(self, context_frame):
        """Context prediction with genre integrated out: the mean prediction over
        every genre the model was trained on, at the caller's fame level.

        Used when the caller does not know the genre. Why not fill the mode:
        this dataset is genre-balanced by construction, so "most common genre"
        is a near-tie decided by luck of the split (it flipped comedy→bluegrass
        between retrains), while genres differ by tens of popularity points at
        fixed fame (16–72 points at 500k listeners). Betting the baseline on an
        arbitrary tie-winner distorts the breakdown; averaging over all genres
        is stable and represents genuine uncertainty instead of a fake specific
        genre.
        """
        genres = sorted(self._known_genres)
        frame = pd.concat([context_frame] * len(genres), ignore_index=True)
        frame["track_genre"] = genres
        return float(self._context_model.predict(frame).mean())

    def frames_for(self, audio_features, context=None):
        """The validated (audio_frame, context_frame) that predict() would build.

        Stage 9 needs the exact rows the models scored in order to attribute over
        them. Rebuilding here — rather than duplicating _frame()'s dtype coercion,
        range handling and missing-feature checks — keeps the explanation and the
        prediction on identical inputs by construction.

        `context_frame` is None when no context was supplied. When the genre is
        unknown we insert the same placeholder predict() uses; the caller is
        expected to marginalize over genres rather than trust that single row.
        """
        self._load()
        audio_frame, _ = self._frame(dict(audio_features), AUDIO_FEATURES, "audio")
        if not context:
            return audio_frame, None

        context_values = dict(context)
        if context_values.get("track_genre") is None:
            context_values["track_genre"] = self._mode_genre
        context_frame, _ = self._frame(context_values, CONTEXT_FEATURES, "context")
        return audio_frame, context_frame

    # ------------------------------------------------------------------ predict
    def predict(self, audio_features, context=None):
        """Score one song.

        audio_features: the 25 librosa descriptors (LIBROSA_FEATURES), as produced
                        by model/audio.py::extract_librosa_features. For an mp3 use
                        predict_from_audio_file(), which extracts them for you.
        context:        {"artists_listeners": int, "track_genre": str | None} or
                        None. track_genre may be None/omitted ("I don't know"):
                        the context prediction is then AVERAGED over every
                        training genre and flagged via Prediction.genre_imputed
                        — an explicit, surfaced fallback, not the pipeline's
                        silent imputer.

        With no context we return the audio contribution only — that is the honest
        answer for an unsigned artist, and it is the part the product is really
        about. Passing fame in additionally yields the composed popularity estimate.
        """
        self._load()

        audio_frame, warnings = self._frame(dict(audio_features), AUDIO_FEATURES, "audio")
        raw_audio = float(self._audio_model.predict(audio_frame)[0])

        prediction = Prediction(
            audio_percentile=None,          # both need the genre; filled in below
            audio_contribution=raw_audio,
            audio_contribution_raw=raw_audio,
            warnings=warnings,
            features=audio_frame.iloc[0].to_dict(),
        )
        # The genre we rank and centre against. Only a genre the caller actually
        # supplied counts: an imputed one is a guess, and ranking a song inside a
        # genre we invented for it would be worse than ranking it globally.
        scoring_genre = None

        if context:
            context_values = dict(context)
            marginalize_genre = context_values.get("track_genre") is None
            if marginalize_genre:
                # "Genre unknown" is a legal input. The placeholder below exists
                # only to satisfy _frame's strict missing-value check (which we
                # keep — it guards against broken extraction upstream); the
                # actual prediction averages over every training genre instead
                # of using it.
                context_values["track_genre"] = self._mode_genre
                prediction.genre_imputed = True

            context_frame, context_warnings = self._frame(
                context_values, CONTEXT_FEATURES, "context"
            )
            genre = context_frame.at[0, "track_genre"]
            if not marginalize_genre and genre not in self._known_genres:
                # Not fatal, but the one-hot block is all zeros, so the context
                # model is extrapolating off a genre it never saw. Say so.
                context_warnings.append(
                    f"track_genre={genre!r} unseen in training; context estimate unreliable"
                )

            if marginalize_genre:
                context_pred = self._marginal_context_prediction(context_frame)
                context_warnings.append(
                    f"track_genre unknown; context averaged over all "
                    f"{len(self._known_genres)} training genres"
                )
            else:
                context_pred = float(self._context_model.predict(context_frame)[0])
                scoring_genre = str(genre)

            # Three-part split (Stage 8). Exactly additive; see _context_split.
            baseline, fame_part, genre_part = self._context_split(context_frame, context_pred)
            if baseline is not None:
                prediction.baseline = baseline
                prediction.fame_contribution = fame_part
                prediction.genre_contribution = genre_part

            prediction.context_contribution = float(
                np.clip(context_pred, POPULARITY_MIN, POPULARITY_MAX)
            )
            # NOTE: composed from the RAW audio score, not the genre-centred one —
            # the genre offset only moves points between the genre and audio bars,
            # it must not change the total.
            prediction.predicted_popularity = float(
                np.clip(context_pred + raw_audio, POPULARITY_MIN, POPULARITY_MAX)
            )
            prediction.context_available = True
            prediction.warnings.extend(context_warnings)
            prediction.features.update(context_frame.iloc[0].to_dict())
            if marginalize_genre:
                # No single genre entered the model — record that, not the
                # validation placeholder, so Stage 9 explains the right thing.
                prediction.features["track_genre"] = (
                    f"(averaged over {len(self._known_genres)} genres)"
                )

        # --- genre-centre the audio score, then rank it -------------------------
        # The raw score is mildly genre-correlated (calibrate.py measures mean audio
        # output from ~-4.5 for romance to ~+1.6 for electro). Move that systematic
        # part onto the genre bar so `audio_contribution` is what THIS recording did
        # relative to its peers. The sum is untouched, so the waterfall still closes.
        offset = self._genre_offset(scoring_genre)
        prediction.audio_contribution = raw_audio - offset
        if prediction.genre_contribution is not None:
            prediction.genre_contribution += offset
        if prediction.context_contribution is not None:
            # Mirror the same shift onto context_contribution. The offset moved
            # from audio's raw score onto the genre bar, and genre is part of
            # context, so context must carry it too — otherwise the legacy
            # two-part identity (context + audio == predicted_popularity) breaks
            # by exactly the offset.
            prediction.context_contribution = float(
                np.clip(prediction.context_contribution + offset, POPULARITY_MIN, POPULARITY_MAX)
            )

        prediction.audio_percentile, prediction.audio_percentile_scope = self._to_percentile(
            raw_audio, scoring_genre
        )
        if prediction.audio_percentile is None:
            prediction.warnings.append(
                f"no {CALIBRATION_FILE}; audio_percentile unavailable. Run: python -m model.calibrate"
            )
        elif prediction.audio_percentile_scope == "all genres" and scoring_genre is not None:
            n = (self._calibration.get("genre_n", {}) or {}).get(scoring_genre)
            prediction.warnings.append(
                f"no per-genre scale for {scoring_genre!r}"
                + (f" (only {n} songs downloaded)" if n else "")
                + "; ranked against all genres instead"
            )

        # numpy/pandas scalars are not JSON-serializable; the API layer would 500.
        prediction.features = {
            k: (v.item() if hasattr(v, "item") else v)
            for k, v in prediction.features.items()
        }
        return prediction

    # ------------------------------------------------------- Stage 6 (Option B)
    def predict_from_audio_file(self, path, context=None):
        """Score an uploaded mp3 end to end: extract librosa descriptors and run
        them through the audio model, which was trained on those same features by
        the same extractor (model/audio.py) — so there is no train/serve skew.

        `context` is the same {"artists_listeners", "track_genre"} dict as
        predict(); omit it to get the audio contribution alone.
        """
        from model.audio import extract_librosa_features  # heavy import (librosa)

        features = extract_librosa_features(path)
        return self.predict(features, context=context)
