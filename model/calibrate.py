"""Build the artifact that turns a raw residual into a 0-100 audio percentile.

    python -m model.calibrate
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from model.features import (
    AUDIO_FEATURES,
    AUDIO_NUMERIC_FEATURES,
    AUDIO_CATEGORICAL_FEATURES,
    CONTEXT_FEATURES,
    CONTEXT_NUMERIC_FEATURES,
    CONTEXT_CATEGORICAL_FEATURES,
)
from model.predictor import ARTIFACT_DIR, CALIBRATION_FILE
from model.train import (
    DEFAULT_LGBM_PARAMS,
    GROUP_COLUMN,
    TARGET,
    load_librosa_training_set,
    make_oof_predictions,
)

N_QUANTILES = 101  # 0th..100th percentile inclusive
# Below this many downloaded songs, a per-genre grid is noise (a 101-point quantile
# curve fitted to a dozen samples), so that genre falls back to the global scale.
MIN_GENRE_N = 30


def build_calibration(df=None, n_splits=5, params=None):
    # Option B: calibrate on the SAME downloaded librosa set the audio model was
    # trained on — its out-of-fold residual spread is what real uploads produce.
    df = load_librosa_training_set() if df is None else df.copy()
    params = params or DEFAULT_LGBM_PARAMS
    df = df.dropna(
        subset=[TARGET, GROUP_COLUMN] + CONTEXT_FEATURES + AUDIO_FEATURES
    ).reset_index(drop=True)

    print(f"Calibrating on {len(df)} librosa rows / {df[GROUP_COLUMN].nunique()} artists")

    # 1) fame -> popularity, out-of-fold
    context_oof, _ = make_oof_predictions(
        df[CONTEXT_FEATURES], df[TARGET], df[GROUP_COLUMN],
        best_params=params,
        nfeatures=CONTEXT_NUMERIC_FEATURES,
        cfeatures=CONTEXT_CATEGORICAL_FEATURES,
        n_splits=n_splits,
    )
    residual = pd.Series(df[TARGET].to_numpy() - context_oof, index=df.index)

    # 2) audio -> residual, out-of-fold. These predictions are the calibration grid.
    audio_oof, _ = make_oof_predictions(
        df[AUDIO_FEATURES], residual, df[GROUP_COLUMN],
        best_params=params,
        nfeatures=AUDIO_NUMERIC_FEATURES,
        cfeatures=AUDIO_CATEGORICAL_FEATURES,
        n_splits=n_splits,
    )

    quantiles = np.quantile(audio_oof, np.linspace(0, 1, N_QUANTILES))
    spearman = float(spearmanr(residual, audio_oof)[0])

    # --- per-genre grids -----------------------------------------------------
    # The product ranks a song against ITS OWN genre: "beats 73% of pop tracks" is
    # the comparison a user can act on, and it is apples-to-apples in a way the
    # global scale is not. Genres with too few downloaded songs get no grid and
    # fall back to global at serving (flagged), rather than shipping a noisy one.
    audio_series = pd.Series(audio_oof, index=df.index)
    per_genre, genre_n, genre_offset = {}, {}, {}
    for genre, idx in df.groupby("track_genre").groups.items():
        values = audio_series.loc[idx].to_numpy()
        genre_n[str(genre)] = int(len(values))
        if len(values) >= MIN_GENRE_N:
            per_genre[str(genre)] = [
                float(q) for q in np.quantile(values, np.linspace(0, 1, N_QUANTILES))
            ]
            # The audio model's output is still mildly genre-correlated even though
            # the residual it predicts had genre removed — the context model's
            # removal is imperfect, so the audio model re-captures a little of it.
            # Measured: mean audio contribution runs from ~-4.5 (romance) to ~+1.6
            # (electro), i.e. 36% of the total spread. Serving subtracts this offset
            # so `craft` is a within-genre deviation and the systematic part is
            # attributed to the genre bar, where it belongs.
            genre_offset[str(genre)] = float(values.mean())
    print(f"per-genre grids: {len(per_genre)} of {len(genre_n)} genres "
          f"(>= {MIN_GENRE_N} songs); the rest fall back to the global scale")
    if genre_offset:
        lo = min(genre_offset, key=genre_offset.get)
        hi = max(genre_offset, key=genre_offset.get)
        print(f"genre offsets: {lo} {genre_offset[lo]:+.2f} .. {hi} {genre_offset[hi]:+.2f}")

    # Reference fame: the baseline of the serving waterfall is "a median-fame track
    # of an average genre", so predictor.py needs the fame the base was taken at.
    reference_fame = float(df["artists_listeners"].median())

    calibration = {
        "quantiles": [float(q) for q in quantiles],
        "per_genre": per_genre,
        "genre_offset": genre_offset,
        "genre_n": genre_n,
        "min_genre_n": MIN_GENRE_N,
        "reference_fame": reference_fame,
        # Recorded so the score can be reported honestly downstream: this is how
        # well the ranking we are calibrating actually ranks.
        "oof_residual_spearman": spearman,
        "residual_std": float(residual.std()),
        "predicted_residual_std": float(np.std(audio_oof)),
        "n_rows": int(len(df)),
        "n_artists": int(df[GROUP_COLUMN].nunique()),
    }

    path = ARTIFACT_DIR / CALIBRATION_FILE
    path.write_text(json.dumps(calibration, indent=2))

    print(f"OOF residual Spearman: {spearman:.4f}  (CLAUDE.md floor: 0.15)")
    print(f"true residual std {calibration['residual_std']:.2f} vs "
          f"predicted std {calibration['predicted_residual_std']:.2f} "
          "(predictions are narrower — that is why we rank preds against preds)")
    print(f"Wrote {path}")
    return calibration


if __name__ == "__main__":
    build_calibration()
