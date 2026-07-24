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

# --- librosa descriptors : the SERVING audio model's inputs (Stage 6 Option B) ---
# Mirrors model/audio.py::extract_librosa_features — the one extractor training and
# the backend both call. The audio model consumes these because an uploaded mp3 can
# only yield librosa descriptors, never Spotify's engineered features.
LIBROSA_FEATURES = [
    # rhythm
    "lb_tempo",
    "lb_onset_rate",
    # energy / dynamics
    "lb_rms_mean",
    "lb_rms_std",
    "lb_dynamic_range",
    # spectral shape — mean AND spread. The mean says how bright/noisy the song is
    # on average; the std says how much it moves. A dynamic arrangement and a flat
    # loop can share a mean and differ entirely in spread.
    *[
        f"lb_{name}_{stat}"
        for name in ("centroid", "bandwidth", "rolloff", "flatness", "contrast", "zcr")
        for stat in ("mean", "std")
    ],
    # timbre — per-coefficient mean, spread, and delta-spread (how fast timbre moves)
    *[
        f"lb_mfcc{i}_{stat}"
        for i in range(1, 14)
        for stat in ("mean", "std", "delta_std")
    ],
    # tonal
    "lb_chroma_mean",
    "lb_chroma_std",
]

# --- Audio model : predicts the popularity RESIDUAL (the song's contribution) ---
# AUDIO_* are the serving audio model's inputs. Under Option B that IS the librosa
# set (all continuous, no categoricals — key/mode/time_signature are Spotify-only).
AUDIO_NUMERIC_FEATURES = list(LIBROSA_FEATURES)
AUDIO_CATEGORICAL_FEATURES = []
AUDIO_FEATURES = AUDIO_NUMERIC_FEATURES + AUDIO_CATEGORICAL_FEATURES

# --- Spotify audio features : RESEARCH track only (notebooks), never served ---
# The documented 66k finding (residual Spearman ≈0.18–0.20) uses these; the serving
# path cannot, since an uploaded mp3 does not come with Spotify's features.
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

# The models were fit on a DataFrame where key/mode/time_signature were int64 and
# track_genre was str. OneHotEncoder(handle_unknown="ignore") learned *integer*
# categories, so a JSON payload sending key="2" instead of 2 does NOT raise — it
# encodes to an all-zero block and the prediction is quietly wrong. Coercing every
# incoming field to the training dtype is what stops that.
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

# The serving audio inputs (librosa descriptors) are all continuous floats. Added
# programmatically so this can never fall out of sync with LIBROSA_FEATURES.
FEATURE_DTYPES.update({name: "float64" for name in LIBROSA_FEATURES})

# Ranges Spotify guarantees for its audio features. Used to reject out-of-domain
# input rather than let a tree extrapolate off the end of its training range.
# NOTE: librosa descriptors are deliberately absent here — they are computed by our
# own extractor (not a user-supplied JSON payload, the case this guard was built
# for) and have no fixed natural bounds, so the predictor leaves them unclipped.
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
