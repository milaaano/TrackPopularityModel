"""Tests for the FastAPI serving layer.

These check the HTTP contract the frontend already codes against — field names,
status codes, and the additivity of the returned breakdown. They do not re-test
model quality (train.py's metrics do that) or the predictor's internals
(test_predictor.py does that).

    /opt/anaconda3/envs/ml/bin/python -m pytest tests/test_backend.py -q
"""

import glob
import io

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


@pytest.fixture(scope="module")
def mp3_path():
    files = sorted(glob.glob("data/audio/*.mp3"))
    if not files:
        pytest.skip("no downloaded mp3s to score")
    return files[0]


def _post(mp3_path, **overrides):
    fields = {
        "artist_name": "Radiohead",
        "genre": "alternative",
        "explicit": "false",
        **overrides,
    }
    with open(mp3_path, "rb") as handle:
        return client.post(
            "/analyze",
            files={"audio_file": ("track.mp3", handle, "audio/mpeg")},
            data=fields,
        )


def test_health_is_cheap_and_ok():
    # The frontend polls this to wake a sleeping host, so it must not load models.
    assert client.get("/health").status_code == 200
    assert client.get("/health").json()["status"] == "ok"


def test_analyze_returns_the_full_contract(mp3_path):
    response = _post(mp3_path)
    assert response.status_code == 200, response.text
    body = response.json()
    for field in (
        "predicted_popularity", "baseline", "fame_contribution",
        "genre_contribution", "audio_contribution", "audio_percentile",
        "audio_percentile_scope", "fame_estimated", "genre_imputed",
        "shap_context", "shap_audio", "warnings", "explanation",
        "explanation_source", "audio_features",
    ):
        assert field in body, f"missing {field}"
    assert body["explanation_source"] in {"llm", "template"}
    assert isinstance(body["explanation"], str) and body["explanation"]


def test_breakdown_is_additive(mp3_path):
    """The product claim, over HTTP: the four parts must sum to the total.
    Same identity test_predictor.py pins in-process — asserted again here
    because serialization is where a field can quietly go missing."""
    body = _post(mp3_path).json()
    total = (
        body["baseline"] + body["fame_contribution"]
        + body["genre_contribution"] + body["audio_contribution"]
    )
    assert total == pytest.approx(body["predicted_popularity"], abs=0.05)


def test_shap_context_matches_the_waterfall(mp3_path):
    """The driver bars and the tiles must never disagree — shap_context is
    derived from the same split that produced the tiles, not a second pass."""
    body = _post(mp3_path).json()
    assert body["shap_context"]["fame"] == pytest.approx(body["fame_contribution"])
    assert body["shap_context"]["genre"] == pytest.approx(body["genre_contribution"])


def test_unknown_genre_marginalizes_and_flags(mp3_path):
    body = _post(mp3_path, genre="").json()
    assert body["genre_imputed"] is True
    assert any("averaged over all" in w for w in body["warnings"])
    # The LLM-facing caveat must exist so the explanation can state it.
    assert any("No genre was given" in c for c in body["caveats"])


def test_known_genre_produces_no_caveats(mp3_path):
    """The Bug-1 guard at the API boundary: a supplied genre and a known artist
    leave `caveats` empty, so nothing downstream can claim otherwise. Previously
    the LLM received `genre_imputed: false` and read it as 'could not be imputed'."""
    body = _post(mp3_path, genre="pop", artist_name="Radiohead").json()
    assert body["genre"] == "pop"
    assert body["caveats"] == []
    assert body["genre_imputed"] is False
    # And the shipped text must not contradict that.
    lowered = body["explanation"].lower()
    for phrase in ("genre was unknown", "could not be imputed", "not found"):
        assert phrase not in lowered, f"explanation invented a caveat: {phrase!r}"


def test_audio_standing_states_the_direction(mp3_path):
    """The standing is pre-computed server-side so nothing downstream has to work
    out whether a percentile is good — the model got that backwards once."""
    body = _post(mp3_path).json()
    pct, standing = body["audio_percentile"], body["audio_standing"]
    assert standing, "audio_standing must be present whenever a percentile exists"
    expected = "above average" if pct >= 50 else "below average"
    assert standing.startswith(expected), f"{pct=} but standing={standing!r}"
    # The raw fields stay in the RESPONSE — Results.tsx renders the percentile chip.
    assert body["audio_percentile_scope"]


def test_genre_and_fame_standing_state_the_correct_direction(mp3_path):
    """Generalizes the audio_standing fix: genre_contribution was once narrated
    backwards by the LLM ('+10.4' described as something to subtract). The
    server now states the direction as a fact rather than a bare signed float."""
    body = _post(mp3_path, artist_name="Drake", genre="hip-hop").json()
    gc, fc = body["genre_contribution"], body["fame_contribution"]
    assert body["genre_standing"] is not None
    assert ("ABOVE" in body["genre_standing"]) == (gc >= 0)
    assert ("BELOW" in body["genre_standing"]) == (gc < 0)
    assert ("ABOVE" in body["fame_standing"]) == (fc >= 0)
    assert ("BELOW" in body["fame_standing"]) == (fc < 0)


def test_explanation_does_not_invert_genre_or_fame(mp3_path):
    """Guards the exact observed failure end to end: hip-hop's positive
    contribution narrated as a subtraction, justified by an invented claim."""
    body = _post(mp3_path, artist_name="Drake", genre="hip-hop").json()
    lowered = body["explanation"].lower()
    if body["genre_contribution"] >= 0:
        assert "lower popularity baseline" not in lowered
        assert "subtract" not in lowered


def test_explanation_does_not_leak_field_names(mp3_path):
    body = _post(mp3_path).json()
    lowered = body["explanation"].lower()
    for phrase in ("shap", "caveats listed", "the output", "payload"):
        assert phrase not in lowered, f"explanation leaked a field name: {phrase!r}"


def test_explanation_does_not_invert_the_standing(mp3_path):
    """Guards the exact observed failure end to end."""
    body = _post(mp3_path).json()
    lowered = body["explanation"].lower()
    if body["audio_percentile"] < 50:
        assert "above average" not in lowered
    else:
        assert "below average" not in lowered


def test_explanation_never_predicts_success(mp3_path):
    body = _post(mp3_path).json()
    lowered = body["explanation"].lower()
    for phrase in ("will be popular", "will likely perform", "guaranteed", "is a hit"):
        assert phrase not in lowered, f"forbidden claim in explanation: {phrase!r}"


def test_unknown_artist_falls_back_to_low_prior(mp3_path):
    body = _post(mp3_path, artist_name="zzz_not_a_real_artist_9f3k").json()
    assert body["fame_estimated"] is True
    # The fallback must point LOW (p25 ~11k), never the mean/median.
    assert body["artist_fame"] < 20_000


def test_missing_artist_is_rejected(mp3_path):
    assert _post(mp3_path, artist_name="   ").status_code == 400


def test_unsupported_file_type_is_rejected():
    response = client.post(
        "/analyze",
        files={"audio_file": ("notes.txt", io.BytesIO(b"not audio"), "text/plain")},
        data={"artist_name": "Radiohead", "genre": "pop", "explicit": "false"},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.text


def test_missing_file_is_rejected():
    response = client.post(
        "/analyze",
        data={"artist_name": "Radiohead", "genre": "pop", "explicit": "false"},
    )
    assert response.status_code == 422  # FastAPI's own required-field validation


def test_corrupt_audio_is_reported_not_crashed():
    """A file with the right extension but garbage inside must produce a clean
    4xx, not a 500 — decode failures are user input, not server bugs."""
    response = client.post(
        "/analyze",
        files={"audio_file": ("broken.mp3", io.BytesIO(b"\x00" * 2048), "audio/mpeg")},
        data={"artist_name": "Radiohead", "genre": "pop", "explicit": "false"},
    )
    assert response.status_code in {400, 422}, response.text
