# AI Context: SongAssess Popularity Model

## Current Goal

This repo is building the ML layer for SongAssess. The current modeling design is a residual approach:

1. Train a context model that predicts expected Spotify popularity from artist/context only.
2. Generate out-of-fold context predictions to avoid leakage.
3. Compute residual popularity as `actual_popularity - context_oof_pred`.
4. Train an audio-only model to predict that residual.
5. At inference, return:
   - `context_prediction`
   - `audio_adjustment`
   - `final_prediction = clip(context_prediction + audio_adjustment, 0, 100)`

The purpose is to separate "expected popularity from artist fame and genre" from "what the song/audio features add or subtract."

## Implemented Plan

The implemented plan was:

- Use `data/processed/orig_data.parquet` because it includes `primary_artist`.
- Use artist-grouped validation to reduce leakage:
  - final train/test split uses `GroupShuffleSplit` by `primary_artist`
  - OOF context predictions use `GroupKFold` by `primary_artist`
- Context model features:
  - numeric: `artist_fame_loo`
  - categorical: `track_genre`
- Audio residual model features:
  - numeric: `duration_ms`, `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`
  - categorical/music structure: `key`, `mode`, `time_signature`
- Explicitly excluded from audio residual model:
  - `artist_fame_loo`
  - `track_genre`
  - `explicit`
- Save artifacts:
  - `model/artifacts/context_model.joblib`
  - `model/artifacts/audio_residual_model.joblib`

## Current Code Shape

Main file: `model/train.py`

Important functions/constants:

- `DATA_PATH = data/processed/orig_data.parquet`
- `GROUP_COLUMN = "primary_artist"`
- `CONTEXT_FEATURES = ["artist_fame_loo", "track_genre"]`
- `AUDIO_FEATURES` contains audio/song-structure features only.
- `build_LGBM(best_params=None, nfeatures=None, cfeatures=None)` builds a sklearn pipeline:
  - numeric imputer + scaler
  - categorical imputer + one-hot encoder
  - LightGBM regressor
- `make_oof_predictions(...)` creates group-aware OOF predictions using `GroupKFold`.
- `train_residual_models(...)` trains and saves the final context and audio residual models.

Manual test file: `test.py`

```python
from model.train import train_residual_models

train_residual_models(
    context_params={"n_estimators": 300, "learning_rate": 0.05},
    audio_params={"n_estimators": 300, "learning_rate": 0.05},
    n_splits=10,
)
```

Run with:

```bash
python3 test.py
```

## Current Evaluation Results

The user reported these results from the manual test:

```text
Context MAE: 5.68129367695812
Context R2: 0.7627087322961243
Final MAE: 5.655051655051774
Final R2: 0.7629827845143948
```

Interpretation:

- The audio residual model currently adds very little over the context model.
- MAE improves by about `0.026`, which is small.
- R2 improves by about `0.000274`, also very small.
- This likely means the residual signal from current audio features is weak after `artist_fame_loo + track_genre` explain most predictable popularity.

Additional residual diagnostics were added inside `train_residual_models`:

```text
Zero residual MAE
Audio residual MAE
Audio residual R2
```

These should be checked on the next run. If `Audio residual MAE` is not meaningfully better than `Zero residual MAE`, the audio residual signal is weak or absent with current features.

## Important Notes And Known Issues

- `model/artifacts/context_model.joblib` and `model/artifacts/audio_residual_model.joblib` currently exist.
- `model/predictor.py` is empty.
- `model/__init__.py` currently comments out the predictor import but still has stale `__all__ = ["PopularityPredictor", "train_residual_models"]`.
- If package imports become an issue, fix `model/__init__.py` to export only real symbols or leave it empty.
- Earlier `train_context_model` stacking logic was replaced by `train_residual_models`; do not use the old stacked interpretation.
- `artist_fame_loo` is a very strong feature and may be too close to the target, so it can absorb most explainable signal.
- The current result does not prove audio has no real-world signal; it only says these Spotify-style tabular audio features add almost nothing in this setup.

## Suggested Next Steps

- Rerun `python3 test.py` and inspect the residual diagnostics.
- Add a baseline comparison for an audio model trained directly on popularity to verify whether audio has any raw signal before residualization.
- Consider trying weaker/cleaner context features if `artist_fame_loo` is too target-derived.
- Consider feature engineering or raw-audio features later; current Spotify-style features may not capture enough song quality signal.
- Build a backend predictor only after the model interface is settled:
  - load `context_model.joblib`
  - load `audio_residual_model.joblib`
  - predict context popularity
  - predict audio adjustment
  - return clipped sum
