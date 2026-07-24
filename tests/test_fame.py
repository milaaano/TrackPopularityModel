"""Tests for artist-fame resolution.

Network-free by design: the Last.fm path is either stubbed or disabled, so these
run offline and deterministically. What they actually guard:
  - real DB hits stay real (fame_estimated stays False),
  - unknowns fall to the LOW prior, not the median, and get flagged,
  - name normalization matches the DB keys (the silent-miss trap).

    /opt/anaconda3/envs/ml/bin/python -m pytest tests/test_fame.py -q
"""

import pytest

from model.fame import FameResolver, normalize_artist_name
from model.predictor import SongPredictor


@pytest.fixture(scope="module")
def resolver():
    # No api_key => Last.fm step is skipped, so unknowns deterministically hit the
    # prior with no network access.
    return FameResolver(api_key=None)


def test_known_artist_from_db_is_not_estimated(resolver):
    r = resolver.resolve("Radiohead")
    assert r.source == "database"
    assert r.fame_estimated is False
    assert r.listeners > 1_000_000        # Radiohead is a big Last.fm artist


def test_normalization_is_case_and_space_insensitive(resolver):
    # Same artist, messy input, must resolve to the same DB row.
    a = resolver.resolve("Radiohead")
    b = resolver.resolve("  RADIOHEAD ")
    assert a.listeners == b.listeners == resolver.resolve("radiohead").listeners


def test_unknown_artist_uses_low_prior_not_median(resolver):
    r = resolver.resolve("zzz_not_a_real_artist_9f3k")
    assert r.source == "prior"
    assert r.fame_estimated is True
    # The load-bearing choice: the fallback is the LOW prior (p25 ~11k), well
    # below the median (~50k). If someone "fixes" this to the median, catch it.
    assert r.listeners < 20_000
    assert "prior" in (r.note or "")


def test_normalize_matches_enrichment_contract():
    # NFKC + casefold + collapse whitespace, exactly as the DB was keyed.
    assert normalize_artist_name("  The   Beatles ") == "the beatles"
    assert normalize_artist_name(None) == ""


def test_lastfm_path_used_when_db_misses(monkeypatch):
    # Prove step 2 fires for an unknown artist when a key is present — without a
    # real network call. Stub the private lookup to a known return.
    resolver = FameResolver(api_key="fake-key")
    monkeypatch.setattr(
        resolver, "_lastfm_lookup", lambda name: (777_000, "Some Corrected Name")
    )
    r = resolver.resolve("obscure but on lastfm")
    assert r.source == "lastfm"
    assert r.fame_estimated is False
    assert r.listeners == 777_000
    assert r.matched_name == "Some Corrected Name"


def test_resolve_feeds_predictor(resolver):
    # End-to-end: fame -> context dict -> full breakdown, the real serving path.
    # Audio side is the librosa descriptors the model now consumes (Option B);
    # built from LIBROSA_FEATURES so it never goes stale when that list changes.
    from model.features import LIBROSA_FEATURES

    audio = {name: 0.0 for name in LIBROSA_FEATURES} | {
        "lb_tempo": 123.0, "lb_onset_rate": 3.4, "lb_rms_mean": 0.19,
        "lb_centroid_mean": 2200.0, "lb_bandwidth_mean": 2377.0,
        "lb_rolloff_mean": 4683.0, "lb_contrast_mean": 22.9,
    }
    fame = resolver.resolve("Radiohead")
    predictor = SongPredictor()
    result = predictor.predict(
        audio,
        context={"artists_listeners": fame.listeners, "track_genre": "alternative"},
    )
    assert result.context_available is True
    assert 0 <= result.predicted_popularity <= 100
