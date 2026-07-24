import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "processed" / "orig_data_with_listeners.parquet"
LIBROSA_CACHE = ROOT_DIR / "data" / "audio" / "librosa_features.parquet"
ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "popularity_pipeline.joblib"

TARGET = "popularity"
GROUP_COLUMN = "primary_artist"

from model.features import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    CONTEXT_NUMERIC_FEATURES,
    CONTEXT_CATEGORICAL_FEATURES,
    CONTEXT_FEATURES,
    AUDIO_NUMERIC_FEATURES,
    AUDIO_CATEGORICAL_FEATURES,
    AUDIO_FEATURES,
    LIBROSA_FEATURES,
)

DEFAULT_LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
}

def load_data():
    data = pd.read_parquet(DATA_PATH)

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = data[features]
    y = data[TARGET]

    return X, y

def build_LGBM_pipeline(nfeatures=None, cfeatures=None):
    nfeatures = nfeatures or []
    cfeatures = cfeatures or []

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, nfeatures),
            ("categorical", categorical_pipeline, cfeatures),
        ],
        remainder='drop'
    )

    model = LGBMRegressor(random_state=667, verbose=-1)

    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('model', model),
        ]
    )

    return pipeline

def tune_LGBM_context(X, y):
    param_grid = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__learning_rate": list(np.arange(0.02, 0.13, 0.01)),
    }
    pipeline = build_LGBM_pipeline(
        nfeatures=CONTEXT_NUMERIC_FEATURES,
        cfeatures=CONTEXT_CATEGORICAL_FEATURES,
    )
    gs = RandomizedSearchCV(pipeline, param_grid, scoring='neg_mean_absolute_error', cv=10, n_jobs=-1)
    gs.fit(X, y)

    return {
        key.removeprefix("model__"): value
        for key, value in gs.best_params_.items()
    }

def build_LGBM(best_params=None, nfeatures=None, cfeatures=None):
    best_params = best_params or DEFAULT_LGBM_PARAMS
    model_params = {
        key if key.startswith("model__") else f"model__{key}": value
        for key, value in best_params.items()
    }
    model_params.update({
        "model__random_state": 667,
        "model__n_jobs": -1,
        "model__verbose": -1,
    })

    pipeline = build_LGBM_pipeline(
        nfeatures=nfeatures,
        cfeatures=cfeatures,
    ).set_params(**model_params)

    return pipeline

def make_oof_predictions(
    X,
    y,
    groups,
    best_params=None,
    nfeatures=None,
    cfeatures=None,
    n_splits=10,
):
    group_count = pd.Series(groups).nunique(dropna=False)
    if group_count < 2:
        raise ValueError("Need at least two artist groups for GroupKFold.")

    effective_splits = min(n_splits, group_count)
    group_kfold = GroupKFold(n_splits=effective_splits)

    oof_predictions = np.zeros(len(X))
    fold_models = []

    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X, y, groups=groups), start=1):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]

        X_val_fold = X.iloc[val_idx]

        pipeline = build_LGBM(
            best_params,
            nfeatures=nfeatures,
            cfeatures=cfeatures,
        )
        pipeline.fit(X_train_fold, y_train_fold)

        oof_predictions[val_idx] = pipeline.predict(X_val_fold)
        fold_models.append(pipeline)

        print(f"Finished fold {fold}")

    return oof_predictions, fold_models

def load_librosa_training_set():
    """The audio model's training data: downloaded tracks with
    librosa descriptors, joined to the 66k dataset for popularity + fame + genre +
    artist. Only tracks we have actually downloaded and extracted appear here. (The context model still trains on the 66k.)
    """
    lib = pd.read_parquet(LIBROSA_CACHE)
    full = pd.read_parquet(DATA_PATH).drop_duplicates("spotify_track_id")
    df = lib.merge(full, on="spotify_track_id", how="inner")
    df = df.dropna(
        subset=[TARGET, GROUP_COLUMN] + CONTEXT_FEATURES + LIBROSA_FEATURES
    ).reset_index(drop=True)
    if len(df) < 50:
        raise ValueError(
            f"Only {len(df)} librosa rows joined. Either too few tracks are "
            "downloaded/extracted (Stage 5 — 45 collapsed to fold-means, a few "
            "thousand is the target), or the cache predates the current "
            f"{len(LIBROSA_FEATURES)}-feature extractor and every row was dropped "
            "as incomplete. Re-run the extraction cell in "
            "notebooks/librosa_features.ipynb."
        )
    return df


def _report_audio_metrics(residual, audio_oof, shuffled_oof):
    zero = np.zeros(len(residual))
    print(f"Zero-residual MAE (baseline): {mean_absolute_error(residual, zero):.4f}")
    print(f"Audio residual MAE:          {mean_absolute_error(residual, audio_oof):.4f}")
    print(f"Audio residual R2:           {r2_score(residual, audio_oof):.4f}")
    spearman = spearmanr(residual, audio_oof)[0]
    print(f"Audio residual Spearman:     {spearman:.4f}   (floor 0.15 / good 0.25)")
    print(f"Shuffled-control Spearman:   {spearmanr(residual, shuffled_oof)[0]:.4f}   (noise floor)")
    if spearman < 0.15:
        print("  WARNING: below the Stage-3 floor — do NOT ship this as a points model "
              "(CLAUDE.md Stage 6 gate). Get more/better audio, or serve percentile-only.")
    return spearman

def train_residual_models(context_params=None, audio_params=None, n_splits=10):
    context_params = context_params or DEFAULT_LGBM_PARAMS
    audio_params = audio_params or DEFAULT_LGBM_PARAMS

    # ---- Context model: fame + genre -> popularity, on the full 66k ----------
    full = pd.read_parquet(DATA_PATH)
    full = full.dropna(subset=[TARGET, GROUP_COLUMN] + CONTEXT_FEATURES).reset_index(drop=True)
    print(f"Context: {len(full)} rows / {full[GROUP_COLUMN].nunique()} artists")
    context_model = build_LGBM(
        context_params,
        nfeatures=CONTEXT_NUMERIC_FEATURES,
        cfeatures=CONTEXT_CATEGORICAL_FEATURES,
    )
    context_model.fit(full[CONTEXT_FEATURES], full[TARGET])

    # ---- Audio model: librosa -> residual, on the downloaded set -------------
    audio_df = load_librosa_training_set()
    print(f"Audio: {len(audio_df)} librosa rows / {audio_df[GROUP_COLUMN].nunique()} artists")

    # Leakage-free residual target: OOF context on the downloaded rows.
    context_oof, _ = make_oof_predictions(
        audio_df[CONTEXT_FEATURES], audio_df[TARGET], audio_df[GROUP_COLUMN],
        best_params=context_params,
        nfeatures=CONTEXT_NUMERIC_FEATURES, cfeatures=CONTEXT_CATEGORICAL_FEATURES,
        n_splits=n_splits,
    )
    residual = pd.Series(audio_df[TARGET].to_numpy() - context_oof, index=audio_df.index)

    # OOF evaluation of the audio model + a shuffled-feature noise floor
    # (a permuted target the features provably cannot predict — tells us what a
    # score of "no signal" looks like at this N).
    audio_oof, _ = make_oof_predictions(
        audio_df[AUDIO_FEATURES], residual, audio_df[GROUP_COLUMN],
        best_params=audio_params,
        nfeatures=AUDIO_NUMERIC_FEATURES, cfeatures=AUDIO_CATEGORICAL_FEATURES,
        n_splits=n_splits,
    )
    shuffled_target = pd.Series(
        residual.sample(frac=1.0, random_state=667).to_numpy(), index=residual.index
    )
    shuffled_oof, _ = make_oof_predictions(
        audio_df[AUDIO_FEATURES], shuffled_target, audio_df[GROUP_COLUMN],
        best_params=audio_params,
        nfeatures=AUDIO_NUMERIC_FEATURES, cfeatures=AUDIO_CATEGORICAL_FEATURES,
        n_splits=n_splits,
    )
    _report_audio_metrics(residual, audio_oof, shuffled_oof)

    # Composed sanity check — dominated by fame, so NEVER quote it as audio evidence.
    final_oof = np.clip(context_oof + audio_oof, 0, 100)
    print(f"Final (context+audio) R2:    {r2_score(audio_df[TARGET], final_oof):.4f}  "
          "(mostly fame — not an audio metric)")

    # ---- Final audio model: fit on ALL downloaded rows -----------------------
    audio_model = build_LGBM(
        audio_params,
        nfeatures=AUDIO_NUMERIC_FEATURES,
        cfeatures=AUDIO_CATEGORICAL_FEATURES,
    )
    audio_model.fit(audio_df[AUDIO_FEATURES], residual)

    artifact_dir = MODEL_DIR / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(context_model, artifact_dir / "context_model.joblib")
    joblib.dump(audio_model, artifact_dir / "audio_residual_model.joblib")
    print(f"Saved context_model + audio_residual_model to {artifact_dir}")

    return context_model, audio_model


if __name__ == "__main__":
    train_residual_models()
