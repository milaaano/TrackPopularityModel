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
    audio_percentile: float         # rank among songs OF THE SAME GENRE where possible
    audio_contribution: float       # popularity points, genre-centred ("craft added +2")
    audio_percentile_scope: str | None = None   # the genre it was ranked within, or "all genres"
    audio_contribution_raw: float | None = None  # before the genre offset (diagnostics)

    # --- Context: what fame + genre alone would buy this track ---
    context_contribution: float | None = None
    predicted_popularity: float | None = None   # context + audio, clipped to [0, 100]

    # --- The three-part breakdown (Stage 8) ---------------------------------
    baseline: float | None = None           # a median-fame track of an average genre
    fame_contribution: float | None = None  # what the artist's reach adds
    genre_contribution: float | None = None # what this style adds (incl. the audio genre offset)

    # Whether the context half of the answer can be trusted.
    context_available: bool = False
    genre_imputed: bool = False
    warnings: list[str] = field(default_factory=list)

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
        self._baseline = None

    # ------------------------------------------------------------------ loading
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

        categorical = self._context_model.named_steps["preprocessor"].named_transformers_[
            "categorical"
        ]
        self._known_genres = set(categorical.named_steps["encoder"].categories_[0])

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
        self._load()
        return self._mode_genre

    @property
    def context_model(self):
        self._load()
        return self._context_model

    @property
    def audio_model(self):
        self._load()
        return self._audio_model

    # ------------------------------------------------------------- input hygiene
    def _frame(self, values, columns, kind):
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
        if not self._calibration or genre is None:
            return 0.0
        return float(self._calibration.get("genre_offset", {}).get(genre, 0.0))

    def _context_split(self, context_frame, context_pred):
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
        genres = sorted(self._known_genres)
        frame = pd.concat([context_frame] * len(genres), ignore_index=True)
        frame["track_genre"] = genres
        return float(self._context_model.predict(frame).mean())

    def frames_for(self, audio_features, context=None):
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
        scoring_genre = None

        if context:
            context_values = dict(context)
            marginalize_genre = context_values.get("track_genre") is None
            if marginalize_genre:
                context_values["track_genre"] = self._mode_genre
                prediction.genre_imputed = True

            context_frame, context_warnings = self._frame(
                context_values, CONTEXT_FEATURES, "context"
            )
            genre = context_frame.at[0, "track_genre"]
            if not marginalize_genre and genre not in self._known_genres:
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
        offset = self._genre_offset(scoring_genre)
        prediction.audio_contribution = raw_audio - offset
        if prediction.genre_contribution is not None:
            prediction.genre_contribution += offset
        if prediction.context_contribution is not None:
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
