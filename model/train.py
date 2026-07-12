import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "processed" / "orig_data.parquet"
ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "popularity_pipeline.joblib"

TARGET = "popularity"
from model.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES

GROUP_COLUMN = "primary_artist"

CONTEXT_NUMERIC_FEATURES = ["artist_fame_loo"]
CONTEXT_CATEGORICAL_FEATURES = ["track_genre"]
CONTEXT_FEATURES = CONTEXT_NUMERIC_FEATURES + CONTEXT_CATEGORICAL_FEATURES

AUDIO_NUMERIC_FEATURES = [
    "duration_ms",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]
AUDIO_CATEGORICAL_FEATURES = ["key", "mode", "time_signature"]
AUDIO_FEATURES = AUDIO_NUMERIC_FEATURES + AUDIO_CATEGORICAL_FEATURES

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

def train_residual_models(
    df=None,
    context_params=None,
    audio_params=None,
    n_splits=10,
    test_size=0.2,
):
    df = pd.read_parquet(DATA_PATH) if df is None else df.copy()
    context_params = context_params or DEFAULT_LGBM_PARAMS
    audio_params = audio_params or DEFAULT_LGBM_PARAMS

    required_columns = [TARGET, GROUP_COLUMN] + CONTEXT_FEATURES + AUDIO_FEATURES
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    split = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=667,
    )
    train_idx, test_idx = next(split.split(df, df[TARGET], groups=df[GROUP_COLUMN]))

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    y_train = train_df[TARGET]
    y_test = test_df[TARGET]

    context_oof_preds, context_fold_models = make_oof_predictions(
        train_df[CONTEXT_FEATURES],
        y_train,
        train_df[GROUP_COLUMN],
        best_params=context_params,
        nfeatures=CONTEXT_NUMERIC_FEATURES,
        cfeatures=CONTEXT_CATEGORICAL_FEATURES,
        n_splits=n_splits,
    )

    train_df["context_oof_pred"] = context_oof_preds
    train_df["popularity_residual"] = y_train - context_oof_preds

    context_model = build_LGBM(
        context_params,
        nfeatures=CONTEXT_NUMERIC_FEATURES,
        cfeatures=CONTEXT_CATEGORICAL_FEATURES,
    )
    context_model.fit(train_df[CONTEXT_FEATURES], y_train)

    audio_model = build_LGBM(
        audio_params,
        nfeatures=AUDIO_NUMERIC_FEATURES,
        cfeatures=AUDIO_CATEGORICAL_FEATURES,
    )
    audio_model.fit(train_df[AUDIO_FEATURES], train_df["popularity_residual"])

    context_test_preds = context_model.predict(test_df[CONTEXT_FEATURES])
    audio_adjustments = audio_model.predict(test_df[AUDIO_FEATURES])
    final_preds = np.clip(context_test_preds + audio_adjustments, 0, 100)

    test_residual = y_test - context_test_preds

    print("Zero residual MAE:", mean_absolute_error(
        test_residual,
        np.zeros(len(test_residual)),
    ))
    print("Audio residual MAE:", mean_absolute_error(
        test_residual,
        audio_adjustments,
    ))
    print("Audio residual R2:", r2_score(
        test_residual,
        audio_adjustments,
    ))

    context_test_preds = np.clip(context_test_preds, 0, 100)

    print("Context MAE:", mean_absolute_error(y_test, context_test_preds))
    print("Context R2:", r2_score(y_test, context_test_preds))
    print("Final MAE:", mean_absolute_error(y_test, final_preds))
    print("Final R2:", r2_score(y_test, final_preds))

    artifact_dir = MODEL_DIR / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(context_model, artifact_dir / "context_model.joblib")
    joblib.dump(audio_model, artifact_dir / "audio_residual_model.joblib")

    return context_model, audio_model
