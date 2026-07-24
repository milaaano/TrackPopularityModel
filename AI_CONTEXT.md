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
  - numeric: `artists_listeners` (Last.fm total listeners)
  - categorical: `track_genre`
- Audio residual model features: the 25 librosa descriptors (`LIBROSA_FEATURES`),
  extracted by `model/audio.py` — see CLAUDE.md Stage 6 (Option B).
- Explicitly excluded from **every** model input:
  - `artist_fame_loo` — a leave-one-out target encoding built *from popularity*
    (corr **0.85** with the target, vs 0.41 for `artists_listeners`), and
    uncomputable at serving time for a new upload. It survives only in
    `notebooks/`, where it was used to clean the dataset.
  - `track_genre` — context, not audio; it stays out of the audio model.
  - `explicit`
- Save artifacts:
  - `model/artifacts/context_model.joblib`
  - `model/artifacts/audio_residual_model.joblib`

## Current Code Shape

Main file: `model/train.py`

Important functions/constants:

- `DATA_PATH = data/processed/orig_data.parquet`
- `GROUP_COLUMN = "primary_artist"`
- `CONTEXT_FEATURES = ["artists_listeners", "track_genre"]`
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

## Evaluation Results (SUPERSEDED — leaked)

> **Do not quote these numbers.** They were produced with `artist_fame_loo` in the
> context model, which is a leave-one-out target encoding (corr 0.85 with
> popularity) — so the context R² of 0.76 is largely the target predicting itself.
> With the leak removed (`artists_listeners` + `track_genre`), context R² is ≈0.62.
> Kept only as the record of *why* the feature was dropped. Live numbers: CLAUDE.md §3.

```text
Context MAE: 5.68129367695812
Context R2: 0.7627087322961243     <- leak-inflated
Final MAE: 5.655051655051774
Final R2: 0.7629827845143948
```

Interpretation (as far as it goes):

- The audio residual model added very little over the context model — but against a
  leaked context, "how much audio adds" was not measurable in the first place.
- The honest version of this comparison, on a clean context, is in CLAUDE.md §3.

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
- `artist_fame_loo` is **excluded from all model inputs** (resolved). It is a
  leave-one-out target encoding derived from popularity itself (corr 0.85), and it
  cannot be computed at serving time for a new upload. `artists_listeners` is the
  fame feature everywhere; `artist_fame_loo` remains in `notebooks/` only, as the
  data-cleaning tool it was built to be.
- The current result does not prove audio has no real-world signal; it only says these Spotify-style tabular audio features add almost nothing in this setup.

## Suggested Next Steps

- Rerun `python3 test.py` and inspect the residual diagnostics.
- Add a baseline comparison for an audio model trained directly on popularity to verify whether audio has any raw signal before residualization.
- ~~Consider weaker/cleaner context features~~ — **done**: `artist_fame_loo` was
  replaced by `artists_listeners`, an external Last.fm signal (corr 0.41 with
  popularity rather than 0.85).
- Consider feature engineering or raw-audio features later; current Spotify-style features may not capture enough song quality signal.
- Build a backend predictor only after the model interface is settled:
  - load `context_model.joblib`
  - load `audio_residual_model.joblib`
  - predict context popularity
  - predict audio adjustment
  - return clipped sum
