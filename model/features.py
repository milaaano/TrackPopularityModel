"""Single source of truth for feature names and inference-time dtypes.

train.py and predictor.py MUST read the same lists from here. If a feature list
is defined twice, training and serving drift apart silently — the model still
returns a number, it's just the wrong one.
"""

NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "explicit",
    "key",
    "mode",
    "time_signature",
    "track_genre",
]

# --- Context model (fame) : predicts raw popularity ---
CONTEXT_NUMERIC_FEATURES = ["artists_listeners"]
CONTEXT_CATEGORICAL_FEATURES = ["track_genre"]
CONTEXT_FEATURES = CONTEXT_NUMERIC_FEATURES + CONTEXT_CATEGORICAL_FEATURES

LIBROSA_FEATURES = [
    # rhythm
    "lb_tempo",
    "lb_onset_rate",
    # energy / dynamics
    "lb_rms_mean",
    "lb_rms_std",
    "lb_dynamic_range",
    *[
        f"lb_{name}_{stat}"
        for name in ("centroid", "bandwidth", "rolloff", "flatness", "contrast", "zcr")
        for stat in ("mean", "std")
    ],
    *[
        f"lb_mfcc{i}_{stat}"
        for i in range(1, 14)
        for stat in ("mean", "std", "delta_std")
    ],
    # tonal
    "lb_chroma_mean",
    "lb_chroma_std",
]

AUDIO_NUMERIC_FEATURES = list(LIBROSA_FEATURES)
AUDIO_CATEGORICAL_FEATURES = []
AUDIO_FEATURES = AUDIO_NUMERIC_FEATURES + AUDIO_CATEGORICAL_FEATURES

SPOTIFY_AUDIO_NUMERIC_FEATURES = [
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
SPOTIFY_AUDIO_CATEGORICAL_FEATURES = ["key", "mode", "time_signature"]
SPOTIFY_AUDIO_FEATURES = SPOTIFY_AUDIO_NUMERIC_FEATURES + SPOTIFY_AUDIO_CATEGORICAL_FEATURES

FEATURE_DTYPES = {
    "duration_ms": "float64",
    "danceability": "float64",
    "energy": "float64",
    "loudness": "float64",
    "speechiness": "float64",
    "acousticness": "float64",
    "instrumentalness": "float64",
    "liveness": "float64",
    "valence": "float64",
    "tempo": "float64",
    "artists_listeners": "float64",
    "key": "Int64",
    "mode": "Int64",
    "time_signature": "Int64",
    "track_genre": "object",
}

FEATURE_DTYPES.update({name: "float64" for name in LIBROSA_FEATURES})

FEATURE_RANGES = {
    "danceability": (0.0, 1.0),
    "energy": (0.0, 1.0),
    "speechiness": (0.0, 1.0),
    "acousticness": (0.0, 1.0),
    "instrumentalness": (0.0, 1.0),
    "liveness": (0.0, 1.0),
    "valence": (0.0, 1.0),
    "loudness": (-60.0, 5.0),
    "tempo": (0.0, 250.0),
    "duration_ms": (1000.0, 3_600_000.0),
    "key": (-1, 11),
    "mode": (0, 1),
    "time_signature": (0, 7),
    "artists_listeners": (0.0, 1e9),
}
