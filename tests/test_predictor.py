"""Tests for the serving path.

These do not test whether the model is *good* (train.py's metrics do that). They
test that the backend cannot silently get a wrong-but-plausible number — which is
the failure mode that would never show up as an error in production.

    /opt/anaconda3/envs/ml/bin/python -m pytest tests/test_predictor.py -q
"""

import glob
import json

import pytest

from model.predictor import SongPredictor, PredictorError

# A real median librosa vector from the extracted cache, so the values sit inside
# the distribution the audio model was fit on. Under Stage 6 Option B the audio
# model consumes these 58 librosa descriptors (model/audio.py) — whole-track mean/
# std/delta-std — not Spotify's.
AUDIO = {
    "lb_tempo": 123.0469,
    "lb_onset_rate": 3.2422,
    "lb_rms_mean": 0.203,
    "lb_rms_std": 0.0834,
    "lb_dynamic_range": 0.2771,
    "lb_centroid_mean": 2228.4755,
    "lb_centroid_std": 778.3358,
    "lb_bandwidth_mean": 2381.2562,
    "lb_bandwidth_std": 499.5995,
    "lb_rolloff_mean": 4731.7144,
    "lb_rolloff_std": 1656.2707,
    "lb_flatness_mean": 0.024,
    "lb_flatness_std": 0.0523,
    "lb_contrast_mean": 22.7498,
    "lb_contrast_std": 11.4686,
    "lb_zcr_mean": 0.0933,
    "lb_zcr_std": 0.0534,
    "lb_mfcc1_mean": -93.0668,
    "lb_mfcc1_std": 91.6609,
    "lb_mfcc1_delta_std": 7.672,
    "lb_mfcc2_mean": 92.0035,
    "lb_mfcc2_std": 32.483,
    "lb_mfcc2_delta_std": 4.1775,
    "lb_mfcc3_mean": 0.2295,
    "lb_mfcc3_std": 22.3354,
    "lb_mfcc3_delta_std": 3.0088,
    "lb_mfcc4_mean": 22.6439,
    "lb_mfcc4_std": 16.289,
    "lb_mfcc4_delta_std": 2.1445,
    "lb_mfcc5_mean": 4.1878,
    "lb_mfcc5_std": 13.4241,
    "lb_mfcc5_delta_std": 1.827,
    "lb_mfcc6_mean": 7.2301,
    "lb_mfcc6_std": 11.6413,
    "lb_mfcc6_delta_std": 1.5961,
    "lb_mfcc7_mean": -0.388,
    "lb_mfcc7_std": 10.7723,
    "lb_mfcc7_delta_std": 1.5077,
    "lb_mfcc8_mean": 4.4863,
    "lb_mfcc8_std": 9.7869,
    "lb_mfcc8_delta_std": 1.366,
    "lb_mfcc9_mean": -2.6398,
    "lb_mfcc9_std": 9.4858,
    "lb_mfcc9_delta_std": 1.3262,
    "lb_mfcc10_mean": 3.5917,
    "lb_mfcc10_std": 9.224,
    "lb_mfcc10_delta_std": 1.2885,
    "lb_mfcc11_mean": -2.464,
    "lb_mfcc11_std": 8.7645,
    "lb_mfcc11_delta_std": 1.233,
    "lb_mfcc12_mean": 1.932,
    "lb_mfcc12_std": 8.3271,
    "lb_mfcc12_delta_std": 1.1746,
    "lb_mfcc13_mean": -2.6916,
    "lb_mfcc13_std": 8.1389,
    "lb_mfcc13_delta_std": 1.1563,
    "lb_chroma_mean": 0.3766,
    "lb_chroma_std": 0.2969,
}
CONTEXT = {"artists_listeners": 500_000, "track_genre": "pop"}


@pytest.fixture(scope="module")
def predictor():
    return SongPredictor()


def test_audio_only(predictor):
    result = predictor.predict(AUDIO)
    assert 0 <= result.audio_percentile <= 100
    assert result.context_available is False
    # Without fame we must NOT invent a popularity number.
    assert result.predicted_popularity is None


def test_with_context(predictor):
    result = predictor.predict(AUDIO, context=CONTEXT)
    assert result.context_available is True
    assert 0 <= result.predicted_popularity <= 100
    assert 0 <= result.context_contribution <= 100
    # The whole product claim: final = context + audio (before clipping).
    assert result.predicted_popularity == pytest.approx(
        min(max(result.context_contribution + result.audio_contribution, 0), 100)
    )


def test_librosa_dtype_coercion(predictor):
    """The silent-failure guard. A JSON payload may deliver every number as a
    string; coercion to float64 must make the string and numeric paths identical,
    or the model quietly sees garbage and returns a confident wrong score."""
    as_float = predictor.predict(AUDIO)
    as_str = predictor.predict({k: str(v) for k, v in AUDIO.items()})
    assert as_float.audio_contribution == as_str.audio_contribution


def test_missing_feature_raises(predictor):
    # A dropped feature means upstream extraction broke. Imputing the training
    # median here would score a song we never actually looked at.
    with pytest.raises(PredictorError, match="lb_tempo"):
        predictor.predict({k: v for k, v in AUDIO.items() if k != "lb_tempo"})


def test_librosa_features_not_range_clipped(predictor):
    """librosa descriptors have no fixed natural bounds and come from our own
    extractor (not a user-typed payload), so — unlike the Spotify features — they
    are deliberately NOT clipped. An unusual value passes through untouched."""
    result = predictor.predict({**AUDIO, "lb_centroid_mean": 9999.0})
    assert result.features["lb_centroid_mean"] == 9999.0
    assert not any("lb_centroid_mean" in w for w in result.warnings)


def test_unknown_genre_warns(predictor):
    result = predictor.predict(
        AUDIO, context={"artists_listeners": 1000, "track_genre": "phonk-drift"}
    )
    assert any("unseen in training" in w for w in result.warnings)


def test_genre_none_marginalizes_and_flags(predictor):
    """"I don't know the genre" is a legal input: the context prediction is
    AVERAGED over every training genre and flagged. Not a mode-fill — this
    dataset is genre-balanced, so the "most common" genre is a near-tie won by
    luck, while genres differ by tens of popularity points at fixed fame. The
    average is the honest "I don't know". (Fame is the opposite: absence there
    is informative, so its fill is deliberately low — see test_fame.py.)"""
    result = predictor.predict(
        AUDIO, context={"artists_listeners": 500_000, "track_genre": None}
    )
    assert result.genre_imputed is True
    assert result.context_available is True
    assert 0 <= result.predicted_popularity <= 100
    assert any("averaged over all" in w for w in result.warnings)
    # features must record that no single genre entered the model.
    assert "averaged" in result.features["track_genre"]


def test_marginalized_context_is_the_mean_over_genres(predictor):
    # The semantic contract of genre marginalization — a CONTEXT MODEL property.
    # `context_contribution` also carries each genre's audio offset (mirrored on
    # so context + audio still sums to predicted_popularity — see predict()), so
    # back that out per genre before comparing; the marginalized case uses offset
    # 0 (we don't invent a genre to centre craft against), matching that.
    # (No clipping bites at this fame level, so mean-then-clip equals the plain
    # mean of contributions.)
    per_genre = [
        predictor.predict(
            AUDIO, context={"artists_listeners": 500_000, "track_genre": g}
        ).context_contribution - predictor._genre_offset(g)
        for g in predictor.known_genres
    ]
    marginal = predictor.predict(
        AUDIO, context={"artists_listeners": 500_000, "track_genre": None}
    ).context_contribution
    assert marginal == pytest.approx(sum(per_genre) / len(per_genre), abs=0.05)


def test_four_part_breakdown_sums_to_predicted_popularity(predictor):
    """baseline + fame + genre + craft == predicted_popularity (CLAUDE.md Stage 8).
    Genre carries this genre's audio offset, craft has it subtracted (see predict()
    genre-centring) — the total must not move when points shift between the two."""
    result = predictor.predict(AUDIO, context=CONTEXT)
    assert result.baseline is not None  # calibration must supply reference_fame
    total = (result.baseline + result.fame_contribution
             + result.genre_contribution + result.audio_contribution)
    assert total == pytest.approx(result.predicted_popularity, abs=0.05)


def test_mode_genre_is_a_real_training_genre(predictor):
    assert predictor.mode_genre in predictor.known_genres


def test_genre_key_missing_behaves_like_none(predictor):
    a = predictor.predict(AUDIO, context={"artists_listeners": 500_000, "track_genre": None})
    b = predictor.predict(AUDIO, context={"artists_listeners": 500_000})
    assert b.genre_imputed is True
    assert b.context_contribution == a.context_contribution


def test_known_genre_is_not_flagged(predictor):
    result = predictor.predict(AUDIO, context=CONTEXT)
    assert result.genre_imputed is False
    assert not any("track_genre unknown" in w for w in result.warnings)


def test_output_is_json_serializable(predictor):
    # numpy scalars in `features` would 500 the API layer at json.dumps time.
    json.dumps(predictor.predict(AUDIO, context=CONTEXT).to_dict())


def test_model_actually_uses_the_features(predictor):
    """Two very different songs must not receive the same score.

    LightGBM with too-few rows per leaf collapses to predicting the mean and
    ignores every feature — that exact failure bit us in librosa_features.ipynb
    at N=45. If it ever happens to the shipped audio model, this catches it.
    """
    dim = predictor.predict(
        {**AUDIO, "lb_rms_mean": 0.03, "lb_centroid_mean": 900.0, "lb_onset_rate": 0.8}
    )
    bright = predictor.predict(
        {**AUDIO, "lb_rms_mean": 0.35, "lb_centroid_mean": 4200.0, "lb_onset_rate": 7.5}
    )
    assert dim.audio_contribution != bright.audio_contribution


def test_mp3_path_extracts_and_scores(predictor):
    """Stage 6 Option B: predict_from_audio_file runs the SAME extractor training
    used, end to end. Exercises a real downloaded mp3 (slow — librosa must decode
    and analyze the file)."""
    mp3s = sorted(glob.glob("data/audio/*.mp3"))
    if not mp3s:
        pytest.skip("no downloaded mp3s to score")
    result = predictor.predict_from_audio_file(mp3s[0], context=CONTEXT)
    assert result.audio_contribution is not None
    assert 0 <= result.audio_percentile <= 100
    assert 0 <= result.predicted_popularity <= 100
