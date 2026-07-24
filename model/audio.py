"""Raw-audio → librosa descriptors (Stage 6 input side).

Lifted out of notebooks/librosa_features.ipynb so the notebook, any future
Stage-6 training run, and the backend all extract features with the SAME code.
If extraction drifts between training and serving, the model sees a different
feature distribution than it was fit on and degrades with no error message.

librosa is imported lazily: it pulls in numba/soundfile and costs seconds of
import time, which the feature-payload prediction path should not pay.
"""

import numpy as np

from model.features import LIBROSA_FEATURES

SR = 22050      # analysis sample rate
# Analyze the WHOLE track. The earlier 120s window described only a song's first
# two minutes, so a track that builds, drops, or changes character later looked
# identical to one that never does — exactly the structure we want to measure.
# The cap is a guard against pathological files (DJ sets, hour-long uploads), not
# a analysis window: >99% of real songs are shorter than it.
MAX_DUR = 600


def extract_librosa_features(path, sr=SR, max_dur=MAX_DUR):
    """Return a dict of the LIBROSA_FEATURES descriptors for one audio file.

    Each descriptor is summarized over the whole track. Where a *mean* says "how
    bright/loud/dense is this song on average", the matching *std* says "how much
    does it move" — a song with dynamics and a flat loop can share a mean and
    differ completely in spread, and the mean alone cannot tell them apart.
    """
    import librosa

    y, sr = librosa.load(str(path), sr=sr, mono=True, duration=max_dur)
    f = {}
    duration = len(y) / sr

    # --- rhythm ---
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    f["lb_tempo"] = float(np.atleast_1d(tempo)[0])
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    f["lb_onset_rate"] = len(onsets) / duration

    # --- energy / dynamics ---
    rms = librosa.feature.rms(y=y)[0]
    f["lb_rms_mean"] = float(rms.mean())
    f["lb_rms_std"] = float(rms.std())
    f["lb_dynamic_range"] = float(np.percentile(rms, 95) - np.percentile(rms, 5))

    # --- spectral shape: mean AND spread over the track ---
    for name, values in (
        ("centroid", librosa.feature.spectral_centroid(y=y, sr=sr)),
        ("bandwidth", librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        ("rolloff", librosa.feature.spectral_rolloff(y=y, sr=sr)),
        ("flatness", librosa.feature.spectral_flatness(y=y)),
        ("contrast", librosa.feature.spectral_contrast(y=y, sr=sr)),
        ("zcr", librosa.feature.zero_crossing_rate(y)),
    ):
        f[f"lb_{name}_mean"] = float(values.mean())
        f[f"lb_{name}_std"] = float(values.std())

    # --- timbre: MFCC mean, spread, and delta-spread ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    for i in range(13):
        f[f"lb_mfcc{i + 1}_mean"] = float(mfcc[i].mean())
        f[f"lb_mfcc{i + 1}_std"] = float(mfcc[i].std())
        # The MEAN of a first difference is ~0 by construction (it telescopes), so
        # the informative statistic is its spread: how fast timbre changes.
        f[f"lb_mfcc{i + 1}_delta_std"] = float(mfcc_delta[i].std())

    # --- tonal ---
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    f["lb_chroma_mean"] = float(chroma.mean())
    f["lb_chroma_std"] = float(chroma.std())

    missing = [name for name in LIBROSA_FEATURES if name not in f]
    if missing:
        raise RuntimeError(f"extractor did not produce: {missing}")
    return f
