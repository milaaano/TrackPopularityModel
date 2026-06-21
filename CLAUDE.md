# CLAUDE.md — SoundSignal

## 0. Read this first (how Claude should behave on this project)

**This is an educational project.** The point is not just to ship a working
platform — it is for me (the author) to build real intuition for ML and data
analysis. So the working agreement is:

- **Always explain *why*, not just *what*.** When you suggest a model, a metric,
  a split, a transformation, or a library — explain the reasoning and the
  intuition behind it. Treat every suggestion as a chance to teach.
- **Profound but not excessive.** Go deep enough to build understanding, but
  don't bury the idea in detail. Prefer the load-bearing insight over an
  exhaustive list.
- **Flag assumptions and traps.** If a step has a common pitfall (leakage,
  baseline confusion, a noisy target), say so *before* I hit it.
- **Prefer measuring an assumption over trusting it.** If a method depends on
  something being true ("fame dominates popularity"), propose a quick way to
  *check* that first.
- **Be honest about results.** Low numbers are sometimes the correct outcome.
  Suspiciously high numbers usually mean leakage, not success.

---

## 1. The goal

Two separate questions, two separate models. Keeping them separate is the whole
point — a famous artist can chart with a mediocre song, and a great song by an
unknown can fail to chart. Mixing these into one "popularity" number hides that.

- **Model B — Intrinsic song score:** *How good is the song itself*, after
  removing the advantage of artist fame? (song/audio features only)
- **Model A — Breakout prediction:** *Will this track enter the Spotify Top 200
  within 28 days of release?* (uses **all** features, including fame, momentum,
  competition)

> Note on framing: the upload product pitch is "judge a song from its audio,"
> but Model A is explicitly allowed to use fame and context. B = song-only,
> A = everything. Keep these two framings distinct.

---

## 2. Build order (and why this order)

We build B before A, and models before audio pipeline, because the **main risk
is ML quality, not frontend or plumbing**. Prove signal exists before building
infrastructure around it.

### Stage 1 — Dataset for Model B (+ a fame proxy)
Find a dataset with **audio features + popularity + an artist fame signal**
(followers, or better: artist popularity / monthly listeners / prior chart
entries).
*Why a fame proxy is required:* to later *subtract out* fame, we need fame as a
measurable number. Followers is the cleanest single proxy, but it's crude —
consider adding others if available.

### Stage 2 — EDA / ablation: does fame actually dominate popularity?
Before building anything fancy, measure it. Compare the variance explained (R²)
by:
- fame-only features
- audio-only features
- both combined

*Why:* this is the empirical justification for the entire residual approach. If
fame-only R² is large and audio-only R² is small, the residual trick is
justified. If fame barely matters, we'd skip it. **Never build on an assumption
you haven't measured.**

### Stage 3 — Residualize: train Model B on `popularity_residual`
```
popularity_residual = popularity - predicted_popularity_from_fame_only
```
Then train an **audio-only** model to predict that residual.

*Why:* we're telling the audio model, "don't explain popularity — explain only
the part fame *couldn't* explain." Whatever audio predicts there is song signal
that isn't just fame in disguise.

**TWO MANDATORY GUARDS (these are the easiest ways to fool yourself):**
1. **Out-of-fold residuals.** Compute the fame model's predictions with
   `cross_val_predict`, so each row's residual is based on a prediction made
   *blind* to that row. In-sample residuals are biased toward zero and will
   corrupt Model B's target.
2. **Group-split by artist.** No artist should appear in both train and test.
   Otherwise the model learns artist-shaped residuals — exactly the fame effect
   we tried to remove.

### Stage 4 — Model A: Top-200 breakout (only if B is satisfactory)
Binary target: `entered_top200_28d`. Uses all features (audio, fame, momentum,
release context, competition).
*Why gated on B:* if audio carries no intrinsic signal, the harder chart-entry
task is unlikely to surprise us. (They're somewhat independent, but B is the
cheaper proof-of-signal.)
*Data note:* Model A needs **negative examples** — songs that were released and
did **not** chart. Top-200 data alone is almost all winners and will mislead you.
Use **time-based** validation (train on older, test on newer) because trends
drift.

### Stage 5 — Obtain audio files
*Biggest real-world risk — de-risk it EARLY even though it's built late.*
Spotify deprecated preview URLs and locked down the audio-features API for new
apps, so sourcing audio is harder than it looks. **Before committing to the
pipeline, confirm you can get audio for ~50 tracks.**

### Stage 6 — Estimate Spotify-style features from raw audio
Map librosa-extractable descriptors (tempo, RMS energy, spectral centroid/
contrast, MFCCs, chroma, ZCR, onset density, dynamic range) → Spotify-style
features (danceability, energy, valence, …).

*Open design decision — raise this explicitly:* do we even need this layer?
- **Option A:** estimate Spotify features → interpretable ("danceability"), but
  adds an error-prone layer whose noise propagates into Model B at inference.
- **Option B:** train Model B directly on librosa descriptors → fewer moving
  parts, less interpretable.
Decide consciously, not by default.

### Stage 7 — Backend: upload MP3, extract features, run models
### Stage 8 — Frontend
*Why last:* the value and the risk are in the ML. Frontend is presentation.

### Stage 9 — LLM explanation layer
**No new predictions here** — this stage translates numbers that A and B
already produced into language. Two different user-facing questions need two
different grounding sources; conflating them is the main way this stage goes
wrong.

**"Why did/might this song become popular?"**
Grounded in **per-instance feature attribution** (e.g. SHAP values) for *this
specific song* from Models A and B.
*Why SHAP, not the global `feature_importances_`:* global importance says
"energy matters across the dataset in general." SHAP says "for THIS song, high
energy contributed +5 to its score, low danceability cost it −3." The question
is about *this song*, so the explanation needs to be instance-level too.

**"What could be added to make it more popular?"**
This is a **counterfactual/comparison** question, not a feature-importance one
— importance tells you what matters in general, not what *this song* lacks.
Grounding: compare this song's arrangement/audio features against similar
high-scoring songs (nearest-neighbor within genre/cluster) → concrete deltas
like "your chorus arrives at 0:55; similar hits average 0:28."

> **Dependency flag:** that nearest-neighbor comparison module isn't in stages
> 1–8 yet. Without it, the LLM can only state general patterns ("higher
> beat_density tends to score higher") rather than song-specific suggestions
> ("shorten your intro"). Decide consciously: build a lightweight comparison
> module as part of Stage 9, or ship general-pattern explanations first and
> treat concrete suggestions as a follow-up.

**Guardrails (mirrors "bad use of LLM" in the original spec — non-negotiable):**
- Input to the LLM is **structured numbers only** — SHAP values, residuals,
  comparison deltas, confidence — never a free-form "make this a hit" prompt.
- Every sentence must be traceable to a number in that payload. "The energetic
  intro helped" needs a SHAP value or delta behind it.
- The LLM never asserts the song *will* be popular — only translates scores and
  drivers into plain language, preserving the model's own confidence level.

*Why this is safe to do last, and why it's still risky:* every number it needs
already exists and was already validated by Section 3's metrics — so the LLM's
only job is fluent translation. That's also why hallucination here is the
easiest failure to *miss*: a wrong explanation still reads as confident and
plausible. Spot-check explanations against the underlying SHAP/delta numbers,
the same way you'd spot-check a unit test.

---

## 3. Evaluation

### Model B (intrinsic score) — target is `popularity_residual`
Numbers will be **low by design** — we already removed the easy variance (fame)
before the audio model starts. So judge against a near-zero baseline.

| Level    | Spearman | R²    | Note                                   |
|----------|----------|-------|----------------------------------------|
| Minimum  | > 0.15   | > 0.05| working; clearly above baseline        |
| Good     | > 0.25   | > 0.10| audio carries real signal              |
| Strong   | > 0.35   | —     | **double-check for leakage first**     |

*Why Spearman:* it's rank-based, and the product cares about *ranking* songs by
promise, not predicting an exact number.
*Baselines:* predict-the-mean (R²=0 by definition) and a shuffled-feature
control. A model only matters if it beats these.

### Model A (breakout) — target is `entered_top200_28d`
Main metric: **Precision@K** — the product ranks songs and surfaces the top few,
so what matters is how many of the top-K predictions actually charted.
Also track PR-AUC, ROC-AUC, and a calibration curve.
Beat these baselines: random ranking, artist-history-only, logistic regression.

---

## 4. Standing ML principles for this repo

- **Leakage is the default failure mode.** Out-of-fold for residuals; group-split
  by artist; time-split where time matters. When a result looks too good, suspect
  leakage before celebrating.
- **Baselines before models.** A number means nothing without a dumb baseline to
  beat.
- **Errors propagate in pipelines.** Noisy Stage-6 feature estimates degrade
  Model B at inference even if B trained on clean features.
- **The target is noisy.** Spotify "popularity" is a recency-weighted black box.
  Don't over-trust small differences.

---

## 5. Tech stack (MVP)
Python · pandas · scikit-learn, rest is to be decided

## 6. Out of scope (for now)
Remix generation · chord-by-chord rewriting · real-time TikTok virality ·
"guaranteed hit" claims · full production React frontend · complex LLM advice.
The LLM (Stage 9) only *explains* model outputs — it never predicts, and every
claim must be traceable to a number from A/B (see Stage 9 guardrails).
