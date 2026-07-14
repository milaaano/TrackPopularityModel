from model.train import train_residual_models

train_residual_models(
    context_params={"n_estimators": 300, "learning_rate": 0.03},
    audio_params={"n_estimators": 300, "learning_rate": 0.03},
    n_splits=10,
)