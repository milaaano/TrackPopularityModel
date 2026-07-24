"""Stage 9: turn the numbers the two models produced into plain language.

**No new predictions happen here.** SHAP does the attribution; the LLM only
verbalizes numbers it is handed. That division is the whole safety property:
`base + Σ shap == model output` is an exact identity we check per request, so
every sentence traces to a number that reconciles. If it does not reconcile we
refuse to call the LLM at all and fall back to a deterministic template.

Two glossaries live here:
  - CONTEXT_LABELS  — fame / genre, the context model's two inputs.
  - AUDIO_CONCEPTS  — the 58 librosa descriptors folded into 12 things a human
    can actually picture. Summing SHAP within a group is legitimate (SHAP is
    additive), and it is the only honest way to talk about MFCCs: `mfcc7` has
    no human meaning, so the 39 MFCC columns become one "overall timbre".

    from model.explain import context_shap, audio_shap, generate_explanation
"""

import json
import logging
import os
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

from model.features import LIBROSA_FEATURES

# --- glossaries -------------------------------------------------------------

CONTEXT_LABELS = {"artists_listeners": "fame", "track_genre": "genre"}

# 58 librosa descriptors -> 12 readable concepts. Every raw column appears in
# exactly one group (asserted below), so grouped SHAP still sums to the model
# output and the reconciliation check keeps working.
AUDIO_CONCEPTS = {
    "tempo": ["lb_tempo"],
    "note density": ["lb_onset_rate"],
    "loudness / energy": ["lb_rms_mean", "lb_rms_std"],
    "dynamic range": ["lb_dynamic_range"],
    "brightness": ["lb_centroid_mean", "lb_centroid_std"],
    "tonal spread": ["lb_bandwidth_mean", "lb_bandwidth_std"],
    "high-frequency content": ["lb_rolloff_mean", "lb_rolloff_std"],
    "noisiness": ["lb_flatness_mean", "lb_flatness_std"],
    "timbral contrast": ["lb_contrast_mean", "lb_contrast_std"],
    "percussiveness": ["lb_zcr_mean", "lb_zcr_std"],
    # MFCCs are not individually interpretable — never claim "mfcc7 = warmth".
    "overall timbre": [
        f"lb_mfcc{i}_{stat}"
        for i in range(1, 14)
        for stat in ("mean", "std", "delta_std")
    ],
    "harmonic concentration": ["lb_chroma_mean", "lb_chroma_std"],
}

_grouped = [c for cols in AUDIO_CONCEPTS.values() for c in cols]
assert sorted(_grouped) == sorted(LIBROSA_FEATURES), (
    "AUDIO_CONCEPTS must partition LIBROSA_FEATURES exactly — otherwise grouped "
    "SHAP no longer sums to the model output and the reconciliation check lies."
)

log = logging.getLogger("soundsignal.explain")

RECONCILE_TOL = 0.01  # popularity points


class ReconciliationError(RuntimeError):
    """SHAP values did not sum to the model output — refuse to explain."""


# --- SHAP -------------------------------------------------------------------

def _dense(matrix):
    return matrix.toarray() if sp.issparse(matrix) else matrix


def _explain_pipeline(pipeline, frame):
    """(shap_by_transformed_column, base_value, model_output) for a 1-row frame.

    Our models are sklearn Pipelines (preprocessor -> LGBMRegressor). TreeSHAP
    explains the *tree*, so we transform first and attribute over the columns the
    trees actually saw, then fold those back to input names in the callers.
    """
    import shap  # heavy import (numba); keep it off module load

    pre = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed = _dense(pre.transform(frame))

    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(transformed))[0]
    base = float(np.atleast_1d(explainer.expected_value)[0])
    output = float(model.predict(transformed)[0])
    return dict(zip(pre.get_feature_names_out(), values)), base, output


def _check(base, contributions, output, what):
    total = base + sum(contributions.values())
    if abs(total - output) > RECONCILE_TOL:
        raise ReconciliationError(
            f"{what}: base {base:.4f} + Σshap {sum(contributions.values()):.4f} "
            f"= {total:.4f}, but the model output {output:.4f}"
        )


def context_shap(predictor, context_frame, marginalize_genre=False):
    """Attribute the context prediction to `fame` and `genre`.

    The genre arrives as a one-hot block, so its SHAP is spread across ~114
    columns; summing them back into one number is exact and is what a reader
    means by "what did the genre contribute".

    When the genre is unknown the prediction was marginalized over all genres,
    so the attribution must be too: sweep every training genre and average the
    SHAP values. The base value is constant, so the average still reconciles.
    """
    predictor._load()
    if not marginalize_genre:
        raw, base, output = _explain_pipeline(predictor.context_model, context_frame)
        per_input = _fold_context(raw)
        _check(base, per_input, output, "context SHAP")
        return per_input, base

    genres = sorted(predictor.known_genres)
    sweep = pd.concat([context_frame] * len(genres), ignore_index=True)
    sweep["track_genre"] = genres

    totals, bases, outputs = [], [], []
    for i in range(len(sweep)):
        raw, base, output = _explain_pipeline(predictor.context_model, sweep.iloc[[i]])
        totals.append(_fold_context(raw))
        bases.append(base)
        outputs.append(output)

    averaged = {
        key: float(np.mean([t[key] for t in totals])) for key in totals[0]
    }
    base = float(np.mean(bases))
    _check(base, averaged, float(np.mean(outputs)), "context SHAP (marginalized)")
    return averaged, base


def _fold_context(raw):
    """Transformed column SHAP -> {"fame": x, "genre": y}."""
    folded = {label: 0.0 for label in CONTEXT_LABELS.values()}
    for column, value in raw.items():
        # names look like "numeric__artists_listeners" / "categorical__track_genre_pop"
        name = column.split("__", 1)[-1]
        for source, label in CONTEXT_LABELS.items():
            if name.startswith(source):
                folded[label] += float(value)
                break
    return folded


def audio_shap(predictor, audio_frame):
    """Attribute the audio prediction to the 12 readable concepts."""
    raw, base, output = _explain_pipeline(predictor.audio_model, audio_frame)
    by_feature = {c.split("__", 1)[-1]: float(v) for c, v in raw.items()}

    grouped = {
        concept: float(sum(by_feature.get(c, 0.0) for c in columns))
        for concept, columns in AUDIO_CONCEPTS.items()
    }
    _check(base, grouped, output, "audio SHAP")
    return grouped, base


def top_drivers(contributions, n=6):
    """The n largest-magnitude contributions, biggest first."""
    ordered = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return {k: round(v, 3) for k, v in ordered[:n]}


# --- explanation ------------------------------------------------------------

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:7b")
# R1-style models generate a long reasoning trace before the answer; on CPU that
# is slow. The frontend budgets 150s total for /analyze and librosa+SHAP take only
# a few seconds, so 120 for the LLM leg stays inside that.
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))

SYSTEM_PROMPT = """You explain a song popularity prediction to a musician.

You are given ONLY numbers, already computed. Never predict, never estimate, and
never introduce a number that is not in the payload.

How the score is built (0-100 Spotify-style popularity):
  baseline + fame + genre + craft = predicted_popularity
  - baseline: a typical track of average genre and median artist reach
  - fame: what the artist's audience adds or subtracts
  - genre: what this style is worth, at this artist's level of fame
  - craft: what THIS recording adds beyond other songs of the same genre

Hard rules:
- Every claim must trace to a number you were given.
- The craft part is small by design, usually only a few points. Never dress it
  up as the reason a song will succeed. NEVER claim the song will be popular,
  will perform well, will chart, or is a hit — you are describing a computed
  breakdown, not forecasting an outcome.
- The "caveats" list holds any limitations of this particular result. State each
  one, in your own words, if the list is non-empty. If "caveats" is EMPTY, the
  result has no limitations: do NOT say anything was unknown, missing, not
  found, estimated, imputed, or averaged. The genre you were given in "genre"
  is the real genre of the track — never describe it as unknown.
- The audio drivers are named concepts (e.g. "brightness"). Use those words;
  do not invent acoustic detail you were not given.
- Quote numbers exactly as given, to at most two decimal places. Never write
  more precision than you were shown.
- "audio_standing", "fame_standing" and "genre_standing" already say whether each
  part is above or below average, in words. State them as given. NEVER work out
  for yourself whether a number is good or bad, and NEVER reason about whether a
  genre is "usually" popular, mainstream, or niche in general — you were not
  given that, and the specific number you WERE given is the only truth for this
  song. A positive contribution is ABOVE average and a negative one is BELOW,
  regardless of what you assume about that genre elsewhere.
- Never name a data field ("caveats", "shap", "the output", …). You are writing
  for a musician who has never seen this data, only their song.
- 3-5 sentences, plain language, no bullet points, no headings, no markdown.
"""

# --- output validation ------------------------------------------------------
# The prompt above ASKS for these rules. A small local model does not reliably
# follow them (llama3.2 both invented an "unknown genre" caveat and predicted
# success in the same reply), so the rules are also ENFORCED here. CLAUDE.md
# Stage 9: spot-check sentences against the numbers like unit tests.

_DEGREE_ADVERB = r"(moderately|somewhat|quite|fairly|very|extremely|reasonably|relatively)?\s*"
_FORBIDDEN_CLAIMS = re.compile(
    r"\b("
    r"will (be|likely|probably)?\s*(be\s+)?" + _DEGREE_ADVERB + r"(popular|successful|a hit|chart\w*|perform\w*)"
    r"|likely to (succeed|chart|perform\w*|be\s+" + _DEGREE_ADVERB + r"popular)"
    r"|guaranteed|destined|surefire|sure to (be|become)"
    r"|this (song|track) is a hit"
    r")\b",
    re.IGNORECASE,
)

# 4+ decimal places. The payload is rounded to 2dp before being sent, so a long
# decimal in the reply cannot have come from the input.
_LONG_DECIMAL = re.compile(r"\d+\.\d{4,}")

# Direction claims, checked against the pre-computed `audio_standing`.
_ABOVE_AVERAGE = re.compile(r"\babove (the )?average\b", re.IGNORECASE)
_BELOW_AVERAGE = re.compile(r"\bbelow (the )?average\b", re.IGNORECASE)

# Generalization of the same check to fame/genre: the model separately inverted
# a positive genre_contribution into "we need to subtract 10.4 points", backed by
# an invented claim that "hip-hop has a lower baseline than other genres" — a
# comparison nowhere in the payload. Ground truth here is the ACTUAL SIGN of the
# contribution (already in the payload), not a string we would have to re-parse.
_NEGATIVE_DIRECTION = re.compile(
    r"\b(subtract\w*|lower(ing)?|detract\w*|reduc\w*|decreas\w*|"
    r"pulls?\s+down|drags?\s+down|penal(ty|ize\w*))\b",
    re.IGNORECASE,
)
_POSITIVE_DIRECTION = re.compile(
    r"\b(add\w*|boost\w*|increas\w*|higher|rais\w*)\b", re.IGNORECASE
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _direction_contradiction(text, value, subject_words):
    """True if a sentence mentioning `subject_words` uses direction language
    opposite the actual sign of `value`. Sentence-scoped so an unrelated
    sentence using "reduces" elsewhere (e.g. about craft) cannot trip it."""
    if value is None:
        return False
    positive = value >= 0
    for sentence in _SENTENCE_SPLIT.split(text):
        lowered = sentence.lower()
        if not any(w in lowered for w in subject_words):
            continue
        if positive and _NEGATIVE_DIRECTION.search(sentence):
            return True
        if not positive and _POSITIVE_DIRECTION.search(sentence):
            return True
    return False

# Our JSON schema narrated to a musician ("There are no caveats listed in the
# output", "The Shap values for fame and genre..."). The bare word "caveat" is
# normal English and is NOT banned — the tell is referring to it as a field.
_META_REFERENCES = re.compile(
    r"\b("
    r"shap|payload|json|the output|the input"
    r"|(data|values?|fields?) (provided|given|listed)"
    r"|caveats? (list|listed|field|array|section)"
    r"|no caveats"
    r")\b",
    re.IGNORECASE,
)

# Only rejected when the caveats list is EMPTY — i.e. the model invented a
# limitation that does not apply to this result.
_CAVEAT_LANGUAGE = re.compile(
    r"\b("
    r"not found|unknown genre|genre (was|is) unknown|no genre"
    r"|could not be imputed|was not imputed|averaged across|average across genres"
    r"|not (be )?identified|missing (genre|artist)"
    r")\b",
    re.IGNORECASE,
)


def _round_for_llm(value, digits=2):
    """Round every float in the payload so the model *cannot* quote 15 decimals.

    The model was faithfully echoing its input: json.dumps serializes raw floats
    at full precision, so it received "76.05248677903587" and printed it back.
    Rounding the input removes the source rather than asking the model to format
    — the same reason `caveats` replaced the boolean flags.

    Recursive: shap_context / shap_audio are nested dicts.
    """
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: _round_for_llm(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_for_llm(v, digits) for v in value]
    return value


# Fields the model must not see. `audio_standing` already states the percentile
# AND whether it is good; handing over the bare number as well only gave the model
# something to misjudge — it called the 24th percentile "above average".
_LLM_HIDDEN = ("audio_percentile", "audio_percentile_scope")


def _llm_view(payload):
    """What the model is allowed to see: rounded, minus fields it would only
    misinterpret. The validator still receives the FULL payload, so ground truth
    is never lost — it is only withheld from the thing doing the guessing."""
    return _round_for_llm({k: v for k, v in payload.items() if k not in _LLM_HIDDEN})


def _strip_reasoning(text):
    """Reasoning models (deepseek-r1, …) emit a <think>...</think> block before
    the answer, inside message.content. Keep only what follows the final close
    tag — the reasoning trace must never reach the user OR the validator (it is
    full of "above/below average" musings that would trip the direction checks).

    An UNCLOSED <think> means the answer never arrived (truncated / timed out
    mid-thought), so return "" and let the caller fall back to the template.
    Harmless no-op for non-reasoning models: no tags, nothing stripped.
    """
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return "" if "<think>" in text else text


def _validate_explanation(text, payload):
    """(ok, reason). False => discard the LLM text and serve the template."""
    match = _FORBIDDEN_CLAIMS.search(text)
    if match:
        return False, f"forbidden success claim: {match.group(0)!r}"
    if not payload.get("caveats"):
        match = _CAVEAT_LANGUAGE.search(text)
        if match:
            return False, f"invented caveat with none present: {match.group(0)!r}"
    # The payload is rounded to 2dp before it is sent, so anything longer was
    # not in the input — the model made it up or did arithmetic of its own.
    match = _LONG_DECIMAL.search(text)
    if match:
        return False, f"over-precise number: {match.group(0)!r}"

    # Did it invert the standing? Truth comes from `audio_standing`, the same
    # string the model was given, so both sides read one source.
    # Asymmetric on purpose: a false positive costs a fallback to the template
    # (harmless); a false negative tells the user the opposite of the truth.
    standing = (payload.get("audio_standing") or "").lower()
    if standing.startswith("below") and _ABOVE_AVERAGE.search(text):
        return False, "claims 'above average' but the track is below average"
    if standing.startswith("above") and _BELOW_AVERAGE.search(text):
        return False, "claims 'below average' but the track is above average"

    # Same check, generalized: did the text invert genre's or fame's actual
    # sign? (The observed bug — a positive genre_contribution narrated as a
    # subtraction.) Ground truth is the raw contribution, not a re-parsed string.
    genre_words = {"genre", "style"}
    if payload.get("genre"):
        genre_words.add(str(payload["genre"]).lower())
    if _direction_contradiction(text, payload.get("genre_contribution"), genre_words):
        return False, "genre direction contradicts its actual sign"
    if _direction_contradiction(
        text, payload.get("fame_contribution"), {"fame", "audience", "listeners", "reach"}
    ):
        return False, "fame direction contradicts its actual sign"

    match = _META_REFERENCES.search(text)
    if match:
        return False, f"leaked a data field name: {match.group(0)!r}"
    return True, ""


def _template_explanation(payload):
    """Deterministic fallback — used when Ollama is unreachable or SHAP fails.

    Also the hosted-deploy path: a free Space cannot run a local LLM, so this is
    what serves there. Same numbers, fixed phrasing.
    """
    fame = payload["fame_contribution"]
    genre = payload["genre_contribution"]
    craft = payload["audio_contribution"]
    # Same pre-computed string the model is given, so both paths state the
    # standing identically and neither has to work out the direction itself.
    standing = payload.get("audio_standing")

    parts = [
        f"This track scores {payload['predicted_popularity']:.1f} out of 100. "
        f"Starting from a typical track at {payload['baseline']:.1f}, artist fame "
        f"{'adds' if fame >= 0 else 'removes'} {abs(fame):.1f} and the style "
        f"{'adds' if genre >= 0 else 'removes'} {abs(genre):.1f} points."
    ]
    parts.append(
        f"The recording itself {'adds' if craft >= 0 else 'takes away'} "
        f"{abs(craft):.1f} points"
        + (f" — {standing}." if standing else ".")
    )
    drivers = payload.get("shap_audio") or {}
    if drivers:
        top = sorted(drivers.items(), key=lambda kv: abs(kv[1]), reverse=True)[:2]
        parts.append(
            "The main audio drivers are "
            + " and ".join(f"{name} ({value:+.1f})" for name, value in top)
            + "."
        )
    # Same source of truth the LLM gets: pre-written strings, present only when
    # they actually apply. Nothing here can invent a limitation.
    parts.extend(payload.get("caveats") or [])
    parts.append(
        "Audio typically moves a score by only a few points; fame and genre "
        "dominate popularity, which is why this breakdown keeps them apart."
    )
    return " ".join(parts)


def generate_explanation(payload):
    """(text, source) where source is "llm" or "template".

    Any failure — Ollama not running, timeout, bad response — falls back to the
    template. The caller should never see an exception from here: a missing
    explanation must not fail an otherwise-valid prediction.
    """
    try:
        import requests

        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "options": {"temperature": 0.2},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        # Rounded copy: the model can only quote what it is shown.
                        "content": json.dumps(_llm_view(payload), indent=2),
                    },
                ],
            },
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content") or ""
        text = _strip_reasoning(content).strip()
        if text:
            ok, reason = _validate_explanation(text, payload)
            if ok:
                return text, "llm"
            # The model broke a hard rule. Do not ship a plausible-sounding lie:
            # fall through to the template, which cannot break these rules.
            log.warning("Discarding LLM explanation (%s)", reason)
    except Exception:
        pass  # unreachable, timeout, malformed — all handled the same way
    return _template_explanation(payload), "template"
