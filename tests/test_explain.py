"""Tests for Stage 9 attribution.

The load-bearing property is the identity `base + Σ shap == model output`. It is
what lets an LLM sentence be checked against a number: if the parts do not sum to
the whole, the explanation is describing a model that does not exist. CLAUDE.md
Stage 9 says to refuse to explain in that case, so it is asserted here rather
than assumed.

    /opt/anaconda3/envs/ml/bin/python -m pytest tests/test_explain.py -q
"""

import pandas as pd
import pytest

from model.explain import (
    AUDIO_CONCEPTS,
    ReconciliationError,
    _llm_view,
    _round_for_llm,
    _strip_reasoning,
    _template_explanation,
    _validate_explanation,
    audio_shap,
    context_shap,
    generate_explanation,
    top_drivers,
)
from model.features import AUDIO_FEATURES, LIBROSA_FEATURES
from model.predictor import SongPredictor

CONTEXT = {"artists_listeners": 500_000, "track_genre": "pop"}


@pytest.fixture(scope="module")
def predictor():
    return SongPredictor()


@pytest.fixture(scope="module")
def audio_features():
    # A real median vector, so the row sits inside the training distribution.
    lib = pd.read_parquet("data/audio/librosa_features.parquet")
    return {name: float(lib[name].median()) for name in AUDIO_FEATURES}


@pytest.fixture(scope="module")
def frames(predictor, audio_features):
    return predictor.frames_for(audio_features, CONTEXT)


def test_concepts_partition_the_feature_set():
    """Every librosa column belongs to exactly one concept. If a column were
    missed or double-counted, grouped SHAP would silently stop summing to the
    model output — the reconciliation check would then be checking a lie."""
    grouped = [c for cols in AUDIO_CONCEPTS.values() for c in cols]
    assert sorted(grouped) == sorted(LIBROSA_FEATURES)
    assert len(grouped) == len(set(grouped))  # no column in two groups


def test_context_shap_reconciles(predictor, frames):
    _, context_frame = frames
    contributions, base = context_shap(predictor, context_frame)
    output = float(predictor.context_model.predict(context_frame)[0])
    assert base + sum(contributions.values()) == pytest.approx(output, abs=0.01)
    assert set(contributions) == {"fame", "genre"}


def test_audio_shap_reconciles_after_grouping(predictor, frames):
    audio_frame, _ = frames
    contributions, base = audio_shap(predictor, audio_frame)
    output = float(predictor.audio_model.predict(audio_frame)[0])
    assert base + sum(contributions.values()) == pytest.approx(output, abs=0.01)
    assert set(contributions) == set(AUDIO_CONCEPTS)


def test_marginalized_context_shap_reconciles(predictor, audio_features):
    """Genre unknown: the prediction is averaged over every genre, so the
    attribution must be averaged the same way (CLAUDE.md Stage 9)."""
    _, context_frame = predictor.frames_for(
        audio_features, {"artists_listeners": 500_000, "track_genre": None}
    )
    contributions, base = context_shap(predictor, context_frame, marginalize_genre=True)
    genres = sorted(predictor.known_genres)
    sweep = pd.concat([context_frame] * len(genres), ignore_index=True)
    sweep["track_genre"] = genres
    expected = float(predictor.context_model.predict(sweep).mean())
    assert base + sum(contributions.values()) == pytest.approx(expected, abs=0.01)


def test_reconciliation_failure_is_raised_not_swallowed():
    """The guard must be loud. A silently-wrong attribution is the one failure
    mode a reader cannot detect from the output."""
    from model.explain import _check

    with pytest.raises(ReconciliationError):
        _check(base=10.0, contributions={"a": 1.0}, output=99.0, what="test")


def test_top_drivers_sorts_by_magnitude_not_sign():
    ordered = top_drivers({"a": 0.2, "b": -5.0, "c": 1.0}, n=2)
    assert list(ordered) == ["b", "c"]


def _payload(**overrides):
    base = {
        "predicted_popularity": 74.5,
        "baseline": 38.7,
        "fame_contribution": 3.4,
        "genre_contribution": 30.3,
        "audio_contribution": 2.1,
        "audio_standing": "above average — better than 73% of pop tracks",
        "audio_percentile": 73,
        "audio_percentile_scope": "pop",
        "genre": "pop",
        "caveats": [],
        "shap_audio": {"brightness": 0.6, "overall timbre": 0.8},
    }
    return {**base, **overrides}


_BELOW = {
    "audio_standing": "below average — better than only 24% of hip-hop tracks",
    "audio_percentile": 23.76,
    "audio_percentile_scope": "hip-hop",
}


def test_template_explanation_is_grounded():
    """The fallback must state the real numbers and repeat any caveats verbatim —
    it is what ships whenever the LLM is unavailable or rejected."""
    text = _template_explanation(
        _payload(caveats=["The artist was not found, so fame is a low estimate."])
    )
    assert "74.5" in text and "38.7" in text
    assert "The artist was not found" in text   # caveat surfaced verbatim
    assert "overall timbre" in text             # top driver named


def test_template_quotes_the_precomputed_standing():
    """Both paths must state the standing identically; neither derives it."""
    text = _template_explanation(_payload(**_BELOW))
    assert "below average — better than only 24% of hip-hop tracks" in text
    # And the template can never contradict itself the way the model did.
    ok, _ = _validate_explanation(text, _payload(**_BELOW))
    assert ok


def test_template_invents_no_caveat_when_there_is_none():
    """The bug that started this: claiming the genre was unknown when it wasn't."""
    text = _template_explanation(_payload())
    for phrase in ("not found", "unknown", "imputed", "averaged across"):
        assert phrase not in text.lower(), f"template invented a caveat: {phrase!r}"


def test_validator_rejects_success_claims():
    """llama3.2 ended a real reply with 'will likely perform well'. The prompt
    forbids it; a small model does not reliably obey, so it is enforced."""
    for claim in (
        "This song will likely perform well in terms of popularity.",
        "This track is a hit.",
        "Success is guaranteed for this release.",
    ):
        ok, reason = _validate_explanation(claim, _payload())
        assert not ok, f"should have rejected: {claim!r}"
        assert "forbidden" in reason


def test_validator_rejects_success_claims_with_an_inserted_adverb():
    """The exact gap that let a real reply through: 'likely to be MODERATELY
    popular' evaded the old pattern, which only matched 'be popular' verbatim."""
    ok, reason = _validate_explanation(
        "This means it's likely to be moderately popular on Spotify.", _payload()
    )
    assert not ok and "forbidden" in reason


def test_validator_rejects_genre_direction_inversion():
    """The exact observed bug: genre_contribution is +10.4 (positive — hip-hop
    outperforms an average genre here), but the model narrated it as something
    to subtract, backed by an invented cross-genre popularity claim."""
    text = (
        "However, since this is hip-hop music, which typically has a lower "
        "popularity baseline than other genres, we need to subtract 10.4 points "
        "from that baseline."
    )
    ok, reason = _validate_explanation(
        text, _payload(genre="hip-hop", genre_contribution=10.4)
    )
    assert not ok and "genre direction" in reason


def test_validator_rejects_the_mirrored_genre_inversion():
    ok, reason = _validate_explanation(
        "The genre here boosts the score considerably.",
        _payload(genre="opera", genre_contribution=-8.0),
    )
    assert not ok and "genre direction" in reason


def test_validator_rejects_fame_direction_inversion():
    ok, reason = _validate_explanation(
        "The artist's low audience reach increases the score substantially.",
        _payload(fame_contribution=-5.0),
    )
    assert not ok and "fame direction" in reason


def test_validator_allows_correct_genre_direction():
    """Same shape of sentence as the bug, but describing the TRUE direction —
    must not be caught by an overly broad filter."""
    ok, reason = _validate_explanation(
        "The hip-hop genre performs above an average genre here, adding 10.4 "
        "points at this artist's level of fame.",
        _payload(genre="hip-hop", genre_contribution=10.4),
    )
    assert ok, reason


def test_validator_ignores_direction_words_in_unrelated_sentences():
    """The check is sentence-scoped: craft language using 'reduces' must not
    trip the genre/fame check just because it appears somewhere in the text."""
    text = (
        "The hip-hop genre adds 10.4 points here. Separately, the recording's "
        "craft reduces the score by 1.5 points versus its peers."
    )
    ok, reason = _validate_explanation(
        text, _payload(genre="hip-hop", genre_contribution=10.4)
    )
    assert ok, reason


def test_validator_rejects_invented_caveats_when_none_apply():
    """The exact observed hallucination: 'the genre was unknown' with genre=pop."""
    text = (
        "The score is 74.5. However, since the genre was unknown and could not "
        "be imputed, we use an average style figure across genres."
    )
    ok, reason = _validate_explanation(text, _payload(caveats=[]))
    assert not ok and "invented caveat" in reason


def test_validator_allows_caveat_language_when_it_applies():
    """Same sentence is CORRECT when the caveat genuinely applies — the check
    must be conditional, not a blanket keyword ban."""
    text = "No genre was given, so the style figure is averaged across genres."
    ok, _ = _validate_explanation(text, _payload(caveats=["No genre was given."]))
    assert ok


def test_round_for_llm_flattens_precision_recursively():
    """The model quoted 76.05248677903587 because that is literally what it was
    sent. Round the input and the long decimal has no source."""
    rounded = _round_for_llm(
        {
            "predicted_popularity": 76.05248677903587,
            "baseline": 38.69874798654018,
            "shap_context": {"fame": 22.959569882793417},   # nested dict
            "shap_audio": {"overall timbre": 2.6341234},
        }
    )
    assert rounded["predicted_popularity"] == 76.05
    assert rounded["baseline"] == 38.7
    assert rounded["shap_context"]["fame"] == 22.96      # recursion reached it
    assert rounded["shap_audio"]["overall timbre"] == 2.63


def test_round_for_llm_leaves_non_floats_alone():
    """Strings, lists and None must survive untouched — caveats and the genre
    name are what keep the explanation grounded."""
    payload = _round_for_llm(
        {
            "genre": "hip-hop",
            "caveats": ["No genre was given."],
            "audio_percentile_scope": None,
            "audio_percentile": 73,            # int, not float
        }
    )
    assert payload["genre"] == "hip-hop"
    assert payload["caveats"] == ["No genre was given."]
    assert payload["audio_percentile_scope"] is None
    assert payload["audio_percentile"] == 73


def test_llm_view_hides_the_raw_percentile():
    """The model called the 24th percentile 'above average'. It no longer sees
    the bare number — only `audio_standing`, which states the direction."""
    view = _llm_view(_payload(**_BELOW))
    assert "audio_percentile" not in view
    assert "audio_percentile_scope" not in view
    assert view["audio_standing"].startswith("below average")
    # Everything that keeps the text grounded must survive.
    assert view["genre"] == "pop" and view["caveats"] == []
    assert view["predicted_popularity"] == 74.5


def test_llm_view_still_rounds():
    view = _llm_view(_payload(predicted_popularity=76.05248677903587))
    assert view["predicted_popularity"] == 76.05


def test_strip_reasoning_keeps_only_the_answer():
    """deepseek-r1 emits <think>...</think> before the answer. Only the answer
    may reach the user or the validator — the reasoning trace is full of
    above/below-average musings that would trip the direction checks."""
    text = "<think>The genre is 13.1, so it's above... wait, below?</think>The track scores 74.2."
    assert _strip_reasoning(text) == "The track scores 74.2."


def test_strip_reasoning_drops_truncated_thoughts():
    """An unclosed <think> means the answer never arrived (timed out mid-thought).
    Return "" so generate_explanation falls back to the template cleanly."""
    assert _strip_reasoning("<think>still reasoning and then cut off") == ""


def test_strip_reasoning_is_a_noop_without_tags():
    """Non-reasoning models (llama3.2, phi4-mini) emit no tags — untouched."""
    plain = "This track scores 74.2 out of 100."
    assert _strip_reasoning(plain) == plain


def test_validator_rejects_inverted_standing():
    """The exact observed bug: 23.76th percentile described as above average."""
    text = (
        "The audio percentile value of 23.76 suggests that this track's audio "
        "features are above average for hip-hop songs."
    )
    ok, reason = _validate_explanation(text, _payload(**_BELOW))
    assert not ok and "above average" in reason


def test_validator_rejects_the_opposite_inversion():
    ok, reason = _validate_explanation(
        "Its audio sits below average for the genre.", _payload()
    )
    assert not ok and "below average" in reason


def test_validator_allows_a_matching_direction():
    ok, _ = _validate_explanation(
        "Its audio sits above average for pop tracks.", _payload()
    )
    assert ok


def test_validator_rejects_leaked_field_names():
    """Our JSON schema narrated to a musician."""
    for text in (
        "There are no caveats listed in the output, indicating no limitations.",
        "The Shap values for fame and genre indicate a significant impact.",
        "Based on the data provided, the track scores well.",
    ):
        ok, reason = _validate_explanation(text, _payload())
        assert not ok, f"should have rejected: {text!r}"
        assert "leaked a data field" in reason


def test_validator_allows_caveat_as_ordinary_english():
    """'caveat' is a normal word — only referring to it as a FIELD is the leak."""
    ok, _ = _validate_explanation(
        "One caveat: the artist's reach does most of the work here.", _payload()
    )
    assert ok


def test_validator_rejects_over_precise_numbers():
    text = "The predicted popularity of this track is 76.05248677903587."
    ok, reason = _validate_explanation(text, _payload())
    assert not ok and "over-precise" in reason


def test_validator_allows_two_decimals():
    ok, _ = _validate_explanation("This track scores 76.05, above 38.70.", _payload())
    assert ok


def test_validator_accepts_a_clean_grounded_answer():
    text = (
        "This track scores 74.5 out of 100. A typical track starts at 38.7; the "
        "artist's reach adds 3.4 and the pop style adds 30.3. The recording "
        "itself adds 2.1, driven mostly by its overall timbre and brightness."
    )
    ok, reason = _validate_explanation(text, _payload())
    assert ok, reason


def test_bad_llm_output_falls_back_to_template(monkeypatch):
    """End to end: a rule-breaking LLM reply must never reach the user. It is
    discarded and the template serves, with explanation_source saying so."""
    import model.explain as explain

    class _Resp:
        def raise_for_status(self): pass
        def json(self):
            return {"message": {"content": "This song will be popular for sure."}}

    monkeypatch.setattr(explain, "requests", type("m", (), {"post": staticmethod(lambda *a, **k: _Resp())}), raising=False)
    monkeypatch.setitem(__import__("sys").modules, "requests", type("m", (), {"post": staticmethod(lambda *a, **k: _Resp())}))

    text, source = explain.generate_explanation(_payload())
    assert source == "template"
    assert "will be popular" not in text.lower()


def test_generate_explanation_always_returns_text(predictor, audio_features, monkeypatch):
    """Ollama may not be running. That must degrade to the template, never raise
    — a missing explanation cannot fail an otherwise-valid prediction."""
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:9")  # nothing listens here
    import importlib

    import model.explain as explain

    importlib.reload(explain)
    text, source = explain.generate_explanation(
        {
            "predicted_popularity": 60.0,
            "baseline": 38.0,
            "fame_contribution": 1.0,
            "genre_contribution": 20.0,
            "audio_contribution": 1.0,
            "audio_percentile": 50,
            "audio_percentile_scope": "pop",
            "fame_estimated": False,
            "genre_imputed": False,
            "shap_audio": {"brightness": 0.3},
        }
    )
    assert source == "template"
    assert isinstance(text, str) and len(text) > 40
    importlib.reload(explain)  # restore module state for other tests
