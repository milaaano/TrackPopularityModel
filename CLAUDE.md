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

**Predict a song's Spotify popularity, and show what drove the number** — split
into the part fame bought and the part the song itself earned:

```
predicted_popularity  =  context_contribution  +  audio_contribution
        66            =          62             +        +4
```

Keeping those two parts separate is the whole point. A famous artist can chart
with a mediocre song, and a great song by an unknown can go unheard. A single
blended "popularity" number hides which is which; the decomposition is the
product.

Two models produce the two parts:

- **Context model** — fame (`artists_listeners`) + genre → popularity. This is
  most of the number, and it is also *scaffolding*: its job is to remove fame so
  the audio model gets a clean target (Stage 3).
- **Audio model** — audio features → the popularity **residual** (the part fame
  couldn't explain). This is the "audio_contribution" above, and it is the piece
  this project exists to measure.

> Expect `audio_contribution` to be **small** (typically ±2 to ±4 points).
> That's not a bug — it's the measured finding that fame dominates (Stage 2):
> audio's *unique* contribution, after fame **and genre** are removed, is only
> R²≈0.02. The honesty rule for the whole product: never let the *presentation*
> of the audio part imply more than ±4 of measured effect. See Stage 9.

> **Two tracks (added after the Stage-5/6 work — important):**
> - **Research track** — audio model trained on the **10 Spotify audio features**
>   over the full 66k dataset. Lives in `notebooks/`, and is the *documented ML
>   finding* (fame/genre dominate; audio's unique slice ≈0.02; residual Spearman
>   ≈0.18–0.20). It's the stronger evidence — 66k rows, not a few thousand.
> - **Serving track** — audio model trained on **librosa descriptors** over
>   downloaded audio, because an uploaded mp3 only yields librosa features, never
>   Spotify's. This powers the upload path (Stages 6–7).
> The **context model is identical in both** — it never used audio features.

> **Naming:** we previously called these "Model B" (intrinsic score) and
> "Model A" (breakout). Model A (Top-200 breakout prediction) is **out of scope**
> now — see §6. Refer to the two live models by role: **context model** and
> **audio model**.

---

## 2. Build order (and why this order)

Models before audio pipeline, because the **main risk is ML quality, not
frontend or plumbing**. Prove signal exists before building infrastructure around
it. Stages 1–4 (research) and 7–9 (serving, locally) are **done**; Stage 5
(downloading) is ongoing and Stage 6 has **not cleared its gate** — the audio
model ships percentile-only until it does. Earlier stages are kept because their
*reasoning* still governs every later stage.

> **What "done" means for 7–9:** working and verified **locally** — 70 tests pass
> and a real mp3 goes through the browser end to end. Deployment (Hugging Face
> Space, Vercel, Dockerfile, production CORS) is not built yet.

### Stage 1 — Dataset (+ a fame proxy)  ✅ done
Dataset with **audio features + popularity + an artist fame signal**. We use
`artists_listeners` (Last.fm total listeners) as the fame proxy.
*Why a fame proxy is required:* to later *subtract out* fame, we need fame as a
measurable number.

> **Trap — never feed `artist_fame_loo` to a model.** The dataset also carries a
> leave-one-out target encoding of artist fame. It is **built from popularity
> itself**, and it shows: corr **0.85** with the target, versus **0.41** for
> `artists_listeners`. Using it inflates context R² to ~0.76 (the target predicting
> itself) and makes the audio residual unmeasurable. It is also **uncomputable at
> serving time** — a new upload has no "other songs by this artist" to average.
> Excluded from every model input; it survives only in `notebooks/`, where it was
> legitimately used to *clean* the dataset. Fame in = `artists_listeners`, always.

### Stage 2 — EDA / ablation: does fame actually dominate popularity?  ✅ done
Measured, don't assume. Result: **fame (+genre) explains R²≈0.62; audio's leftover
signal is small** (residual Spearman ≈ 0.18–0.20). A **commonality analysis**
later sharpened *why*: audio alone explains R²≈0.19, but **89% of that is shared
with fame/genre** — and mostly with **genre** specifically (raw
`artists_listeners`↔audio correlations are all ≤0.14). Audio's **unique**
contribution, once fame and genre are removed, is only **R²≈0.02**. That is the
honest size of "what the song earns on its own," and why `audio_contribution` is
small by construction. **Never build on an assumption you haven't measured.**

### Stage 3 — Residualize: train the audio model on `popularity_residual`  ✅ done
```
popularity_residual = popularity - predicted_popularity_from_fame_only
```
The audio model predicts that residual. *Why:* we tell it "don't explain
popularity — explain only the part fame *couldn't*." Whatever it predicts there
is song signal, not fame in disguise.

**THREE MANDATORY GUARDS (the easiest ways to fool yourself):**
1. **Out-of-fold residuals.** The fame model's predictions are made *blind* to
   the row being residualized (`make_oof_predictions`). In-sample residuals are
   biased toward zero and corrupt the audio model's target.
2. **Group-split by artist.** No artist in both train and test — otherwise the
   model learns artist-shaped residuals, exactly the fame effect we removed.
3. **Residualize on the FULL context — fame *and* genre.** Removing fame alone
   leaves genre in the residual, and audio features encode genre
   (acousticness→classical, energy→metal, speechiness→rap). The audio model then
   scores by *re-identifying genre* instead of judging the song. Measured on the
   ~9k downloaded set, dropping `track_genre` from the context model inflates
   audio's residual Spearman from **0.11 → 0.40** — i.e. **≈72% of the apparent
   "audio signal" was genre**.

> **The shuffled-feature control does NOT catch guard 3.** It proves the
> *machinery* is sound (no train/test leakage, no mean-collapse); it cannot tell
> you that you removed the *wrong confounder* — a modeling-definition error, not a
> split error. The detector for guard 3 is a **known-value sanity check**: the
> context model should reproduce R²≈0.60. A context R² of 0.17 means something is
> missing from it, and every downstream "audio" number is inflated.

*(These same guards also build the percentile scale — see Stage 4.)*

### Stage 4 — Score composition + calibration  ✅ done
Serving returns both a raw-points breakdown and a rank:
- **raw points:** `context_contribution + audio_contribution = predicted_popularity`
- **percentile:** where this song's `audio_contribution` ranks among **other songs
  of the same genre** (0–100), falling back to all songs for genres with too few
  downloads. See Stage 8 for why per-genre is not optional.

*Why both (per the user):* raw points are intuitive ("+4"), but +4 alone doesn't
tell you if +4 is good. The percentile does ("+4 = 85th percentile of song
contributions"). *Why the percentile is built by ranking predictions against
other predictions, out-of-fold:* the audio model's predictions span only ≈±2
while true residuals span ≈±11, so ranking predictions against *true* residuals
would squash every song into the middle of the scale. See `model/calibrate.py`.

> **Calibration is per-model.** The percentile scale is built from a *specific*
> audio model's out-of-fold residuals. The librosa serving model (Stage 6) has a
> different residual spread than the Spotify research model, so it needs its
> **own** calibration file — reusing the Spotify one would mis-rank every upload.

### Stage 5 — Obtain audio files  ⏳ in progress (scaling to ~10k)
*Biggest real-world risk.* Spotify deprecated preview URLs and locked down the
API, so we source audio from **YouTube via `yt-dlp`**, searching by the `artists`
+ `track_name` we already have. `notebooks/download.ipynb` does this:

- **Representative sampling** — pick tracks whose audio-feature *distribution*
  matches the full dataset. We use **best-of-random** (draw many random subsets,
  keep the one with the lowest mean per-feature KS distance to the population),
  which beat KMeans/stratified at small N. At large N (≥ a few thousand) a single
  random draw is already representative, so `N_DRAWS` can drop to 1.
- **Duration-filtered download** — reject YouTube hits whose length is off from
  the known `duration_ms` by >20s (cuts remix/cover/live mismatches). Files are
  named by `spotify_track_id`; `manifest.csv` maps id → mp3 → features.
- **Sequential + resumable** — the loop skips ids already on disk. Downloads stay
  sequential on purpose: parallelizing hammers YouTube and trips bot-detection.

> **Trap — `spotify_dl` is dead for this.** It crashes on every track
> (`KeyError: 'genres'`) because Spotify's locked-down API no longer returns the
> artist `genres` field to new apps. We don't need it — the dataset already has
> artist+title, so `yt-dlp` goes straight to YouTube, no Spotify API or creds.

**Goal: ~10k tracks.** 45 was far too few — the audio models collapsed to
per-fold means and couldn't even be measured (`notebooks/librosa_features.ipynb`).

### Stage 6 — Retrain the audio model on librosa features  ⏳ in progress
**Decision: Option B — train the audio model directly on librosa descriptors.**
An upload can only yield librosa features (`model/audio.py`), never Spotify's, so
the serving audio model must speak librosa natively.

*Why not Option A (estimate Spotify features from librosa, then feed the old
model):* it can't add signal — librosa→Spotify is the same information reshaped and
degraded, so if librosa's signal is low the estimated-Spotify score is *lower*,
not higher. Its only advantage was human-readable feature names for Stage 9 — and
we get that instead from a **librosa glossary** (Stage 9). So Option A buys an
error-prone layer for zero benefit; it is now **out of scope** (§6).

**Feature set:** **58 librosa descriptors** (was 25), computed over the **whole
track** (the 600s cap guards pathological uploads, not real songs), by the single
canonical `model/audio.py::extract_librosa_features` — the *same* function training
and the backend both call, so features cannot drift between them:

- **rhythm** — tempo, onset rate; **energy/dynamics** — RMS mean/std, dynamic range
- **spectral shape** — **mean *and* std** for all six descriptors. The mean says how
  bright/noisy a song is; the std says how much it *moves*. A dynamic arrangement
  and a flat loop can share a mean and differ completely in spread.
- **timbre** — per-MFCC **mean, std, and delta-std**. (The *mean* of a first
  difference is ~0 by construction — it telescopes — so the informative statistic is
  its **spread**: how fast timbre changes.)
- **tonal** — chroma mean/std

*Why the whole track, not the first 120s:* the old window was blind to structure — a
song that builds or drops late looked identical to one that never does.

Extraction is **incremental** (only uncached tracks) and **parallel** (one worker
per core), flushed every 200 tracks so a long run is crash-safe. ~2s/track, so a
full 9k re-extraction is ~50 minutes on 6 workers.

> **Trap — the cache is keyed by track id alone.** Change the feature set and every
> cached row still "exists", so extraction silently skips it: you then train on a
> mix of two schemas, or on nothing. The extraction cell carries a **schema guard** —
> any cached row missing a current feature is treated as un-extracted. Note this
> also means changing the feature set takes serving down until re-extract + retrain
> (the saved artifact expects the old width).

> **Gate before serving:** the librosa audio model must clear the Stage-3
> evaluation floor (Spearman > 0.15) on held-out, artist-grouped data *before* it
> is wired into the upload path. Its unique-beyond-genre signal may be as low as
> R²≈0.02 (Stage 2), so failing is a real possibility — if it fails, ship
> percentile-only or say plainly that the audio half is noise. Train → measure
> against the bar → then decide. **Never wire-then-hope.**

**MEASURED (~9k downloaded tracks, fame+genre removed, out-of-fold, artist-grouped):**

| audio features                       | residual Spearman |
|---------------------------------------|-------------------|
| librosa, 25 features (first 120s)     | 0.088–0.112        |
| **librosa, 58 features (whole track)**| **0.120–0.127**    |
| Spotify (10), same ~9k rows           | 0.119              |
| shuffled control                      | ~0.02              |

Three conclusions, all load-bearing:
- **Richer descriptors bought a real, measured gain.** Whole-track analysis plus
  spectral std / MFCC std / MFCC delta-std moved Spearman from 0.088 → 0.120 at
  matched settings (+36% relative). The shuffled control stayed near zero, so this
  is signal, not noise creeping in with more columns.
- **Option B is vindicated, twice over.** The 58-feature librosa model now sits at
  essential parity with the Spotify-10 features on identical rows (0.120 vs 0.119).
  Speaking librosa costs nothing — confirmed again with richer features.
- **The shortfall is still the sample, not the features.** Spotify's own features
  score 0.18–0.20 on the full 66k but only ≈0.12 on this ~9k subset. Since librosa
  has now closed the gap to that same ceiling, further feature engineering has
  diminishing returns — the lever that matters is **more downloaded tracks**.

**Status: below the 0.15 floor → the gate is NOT cleared.** Serve percentile-only.
The most promising path to clearing it is growing the download set further, not
more feature engineering — librosa is no longer the limiting factor.

### Stage 7 — Backend: upload → extract → predict → breakdown  ✅ done (local)
Flow: user uploads an mp3, **types the artist name**, and **picks a genre**
(optional) →
1. **Audio:** `extract_librosa_features` → **librosa audio model** (Stage 6) →
   `audio_contribution`.
2. **Fame:** look up the artist in our DB; if absent, call the **Last.fm API**
   for `listeners`; if still nothing, fall back to a **low fame prior** (25th
   percentile) and set a `fame_estimated` flag. *Implemented in `model/fame.py`.*
   > **Trap — don't fall back to the *mean* or *median*.** An artist in neither
   > our DB nor Last.fm is, by that fact, obscure (Last.fm coverage is huge), so
   > "not found" is *informative* — it points **low**. The mean (~261k,
   > skew-inflated) or median (~50k) would tell an unknown "fame bought you a big
   > chunk" — a lie in the exact half of the breakdown this product exists to keep
   > honest. Use the 25th percentile (~11k) and mark it estimated.
3. **Genre:** the user picks from the **known-genre list**
   (`predictor.known_genres` — free text one-hot-encodes to all zeros and scores
   wrong). If unknown, **marginalize**: predict the context contribution for
   *every* training genre and average, and set a `genre_imputed` flag.
   *Implemented in `model/predictor.py`.*
   > **Trap — don't impute the mode/median genre.** The dataset is genre-balanced
   > by construction, so "most common genre" is a near-tie won by luck (it flipped
   > comedy→bluegrass between fits), while genres differ by *tens* of popularity
   > points at fixed fame (16–72 at 500k listeners). Averaging over all genres
   > ("integrate out what you don't know") is the honest, stable estimate.
   > **Fame vs genre — opposite fills, same principle:** fill *low* when
   > missingness is informative (unknown artist ⇒ obscure), *average* when it is
   > uninformative (unknown genre ⇒ nothing learned about which). Match the fill
   > to what the absence tells you.
   > **Measured — genre CANNOT be inferred from the audio.** Tempting, since it
   > would remove the question from the upload form. But librosa→genre is only
   > **16.7% top-1 / 37.9% top-5** out-of-fold across 114 genres. That is 19× better
   > than random — audio really does carry style information — and still far too
   > weak to act on. What settles it is the *popularity cost*, not the label:
   > substituting the inferred genre mis-states the style contribution by **MAE 8.3
   > points** (a random genre costs 12.0, so inference barely beats guessing), and
   > only **44%** of songs land within 5 points of their true style value. The whole
   > craft signal is ±2–4 points, so the inference error would be **2–4× larger than
   > the thing it sits beside**. Ask the user; marginalize when they don't know.
4. **Compose:** context model → the four-part split; sum → `predicted_popularity`;
   plus the per-genre audio percentile. Implemented in `model/predictor.py`.

**The API** (`backend/app/main.py`, FastAPI — run with
`uvicorn backend.app.main:app --port 8000`):

- `GET /health` — liveness only, **loads no models**. The frontend polls it on
  page load so a sleeping host wakes while the visitor is still reading; that
  only works if it stays cheap.
- `POST /analyze` — multipart `audio_file`, `artist_name`, `genre`, `explicit`.
  Streams the upload to a temp file (librosa needs a real path, and a 25 MB body
  should not sit in memory), **re-validates type and size server-side** (the
  client limit is a convenience, not a guard), and deletes the temp file in a
  `finally`. Returns the four contributions, percentile + scope, both flags,
  grouped SHAP, warnings and the explanation. `explicit` is accepted because the
  frontend sends it, but **no current model consumes it** — silently ignoring it
  beats pretending it matters.
- CORS via an `ALLOWED_ORIGINS` env var (default `http://localhost:3000`), so the
  production origin is a deploy-time setting rather than a code change.

> **Training is offline, never on the backend.** The librosa audio model and its
> calibration are produced by an offline training run and saved as artifacts
> (`*.joblib` / `*.json`); the backend only *loads* them and calls `.predict()`.
> `SongPredictor` and `FameResolver` are module-level singletons — both load
> lazily, and that cost belongs once per process, not once per request.

### Stage 8 — Frontend  ✅ done (local)
*Why last:* the value and the risk are in the ML. Frontend is presentation of the
breakdown.

**Built:** a Next.js 15 / React 19 single page (`frontend/`). `AnalyzeForm.tsx`
takes the file + artist + genre; `Results.tsx` renders the waterfall as five
`.eqtile`s (`baseline + fame + genre + craft = total`) plus the two SHAP driver
lists; `lib/sample.ts` is a clearly-badged canned result so the UI is explorable
when the backend is asleep. The genre `<select>` is generated from
`predictor.known_genres` — a free-text genre would one-hot to all zeros and score
wrong, so the dropdown makes that unreachable.

**DECIDED — a four-part waterfall: baseline + fame + genre + audio-within-genre.** Genre is a
property of the song, not something fame bought, so it moves to the song's side of
the ledger. The split comes from the context model directly — **no SHAP at serving**,
because marginalizing gives an exactly-additive decomposition for free:

- `baseline` = the marginal context at a **reference fame** (median), averaged over
  every genre — "a median-fame track of an average genre"
- `fame` = marginal context at *this* fame − baseline
- `genre` = *this* genre's context − marginal at this fame
- `craft` = audio model output − that genre's mean output (see the trap below)

All four sum exactly to the prediction. Implemented in `model/predictor.py`
(`_context_split`); when genre is unknown we marginalize anyway and the genre term
falls out to ~0 on its own, no special case.

```
typical track 38.7  +  artist fame +3.4  +  style (genre) +30.3  +  craft +2.1  =  74.5
                       \__ external __/     \_________ what the song is _________/
```

*Why split genre out at all:* a two-part split forces a choice between an
honest-but-tiny audio number and a big-but-misleading one that is mostly genre.
Separating them shows both, correctly labelled — "pop averages +30; within pop this
recording adds +2" — and stops genre from masquerading as craft. A waterfall (not a
stacked bar) because contributions can be **negative**.

**The audio percentile ranks within the song's OWN genre.** Measured on ~9k songs,
the globally-median audio score lands anywhere from the **15th to the 78th
percentile** depending on genre — a global rank would tell a classical artist they
are terrible and a pop artist they are fine for identical work. Genres with fewer
than 30 downloaded songs get no grid and fall back to the global scale, flagged.

> **Trap — the raw audio score is still mildly genre-correlated.** The residual had
> genre removed, but the context model's removal is imperfect, so the audio model
> re-captures a little of it: mean audio output runs from **−4.5 (romance) to +1.6
> (electro)**, about **36% of its total spread**. Serving subtracts that per-genre
> offset from `audio_contribution` and adds it to the genre bar. The total is
> unchanged (the waterfall still closes), but `craft` stops penalising a classical
> track for being classical, and it now agrees with the within-genre percentile
> instead of contradicting it.

*Presentation caveats:* style is the largest term, so the genre input must stay
**user-editable** (it cannot be inferred — Stage 7); phrase it category-level ("pop
tracks average +30"), never "your song's pop-ness earned +30"; and the headline for
the user is the **craft percentile** — the only part they control and the only part
comparable across genres.

### Stage 9 — LLM explanation layer  ✅ done (local)
**No new predictions here** — this stage only turns the numbers the two models
already produced into plain language. Implemented in `model/explain.py`.

*Division of labor — SHAP attributes, the LLM only verbalizes.* SHAP does the
breakdown (`base + Σ shap = model output`, an exact identity unit-tested per
request in `tests/test_explain.py`). The LLM never decides what drove the score; it
translates already-computed numbers into words.

**Audio: 58 descriptors → 12 readable concepts.** `AUDIO_CONCEPTS` groups the raw
columns into things a person can picture — tempo, note density, loudness/energy,
dynamic range, brightness, tonal spread, high-frequency content, noisiness, timbral
contrast, percussiveness, **overall timbre**, harmonic concentration. Summing SHAP
within a group is legitimate (SHAP is additive) and is the only honest way to talk
about MFCCs: `mfcc7` has no human meaning, so all 39 MFCC columns become one
"overall timbre". A module-level assert enforces that the groups **partition**
`LIBROSA_FEATURES` exactly — miss or double-count a column and grouped SHAP quietly
stops summing to the model output, which would make the reconciliation check a lie.

> **Trap — SHAP and the Stage-8 waterfall are different, equally valid splits.**
> Running `TreeExplainer` on the *context* model gives a fame/genre division that
> does **not** match `_context_split`, because the baselines differ: SHAP starts
> from `expected_value` (≈38.8, the mean training prediction) while the waterfall
> starts from `marginal(median fame)` (≈38.2). At 500k/pop the waterfall says
> fame +5.6 / genre +29.3; SHAP says +4.2 / +30.1. Both sum to the same 73.1.
> Worse, when the genre is marginalized the waterfall's genre term is exactly **0**
> while averaged SHAP is **≈+2.0**. Showing both would put two different "genre"
> numbers on one screen. **So serving derives `shap_context` from the waterfall
> split, not from a second SHAP pass** — the context model has only two inputs, so
> the split already *is* its complete attribution. Real SHAP is used only for
> audio, where 58 features genuinely need collapsing. `context_shap()` stays in
> `explain.py` for research use; `tests/test_backend.py` pins that the tiles and
> the driver bars can never disagree.

**The LLM is local Ollama** (`localhost:11434`, model via `OLLAMA_MODEL`, default
`llama3.2`), called over plain HTTP with `requests` — no SDK, no API key. The
system prompt states the arithmetic and the guardrails below.

> **Trap — negated booleans get misread.** Serving used to send
> `genre_imputed: false` straight to the model. `llama3.2` read the negated field
> name as "genre could **not** be imputed" and invented an "unknown genre" caveat
> for a track whose genre was known and used — while correctly reporting every
> other number in the same reply. **Fixed:** booleans were replaced by a
> `caveats: []` list of pre-written sentences, populated only when a flag is true.
> An empty list has nothing to negate and nothing to misread.

> **Trap — bare signed numbers get reinterpreted, sometimes backwards.** Measured
> twice: a raw 23.76th percentile was narrated as "above average" (it was below),
> and a positive `genre_contribution` of +10.4 was narrated as "we need to
> subtract 10.4 points," backed by an invented claim — "hip-hop typically has a
> lower popularity baseline than other genres" — that appears nowhere in the
> payload. A bare number lets a small model's own genre-popularity priors
> override the sign it was actually given. **Fixed:** pre-computed
> `audio_standing` / `genre_standing` / `fame_standing` fact-strings state the
> direction in words ("below average — better than only 24% of hip-hop tracks");
> the ambiguous raw percentile is hidden from the model's view entirely
> (`_llm_view` in `model/explain.py`). The point values for fame/genre/audio stay
> visible, since they still need to be quoted — only the field that was actually
> misread is withheld.

> **Trap — prompt-only rules do not reliably hold at this model size; enforce,
> don't just instruct.** Even handed the correct stated direction, `llama3.2`
> inverted the audio-percentile direction in **every one of 3 live test runs** —
> giving it the right fact reduced the error but did not eliminate it. Numbers
> are also rounded to 2dp before being sent (`_round_for_llm`): the model was
> quoting 15-decimal floats verbatim, because `json.dumps` serializes full float
> precision and it simply echoes what it is shown. **`_validate_explanation()`**
> checks the raw reply against ground truth already in the payload — forbidden
> success-claim language ("will be popular," tolerant of inserted adverbs like
> "moderately"), invented caveat language when `caveats` is empty, over-precise
> numbers, direction contradictions (checked against the actual signs in the
> payload, not a re-parsed string, and scoped per-sentence so unrelated wording
> can't false-positive), and leaked schema/field names ("shap," "caveats
> listed"). Any violation discards the reply and falls back to the template.

> **Consequence, measured:** because the audio-direction check alone rejected 3
> of 3 live runs, `explanation_source` reads `"template"` often in practice —
> not just during rare Ollama outages. This is not a regression: the template is
> correct by construction (CLAUDE.md's honesty bar applies to it exactly as much
> as to the LLM), and it was always the fallback. The real calibration is that at
> `llama3.2`'s size, the LLM narration is a **bonus layered on a reliable
> template**, not the reverse — never assume the "real LLM" path is the common
> case without measuring `explanation_source` across real requests.

> **Any failure falls back to a deterministic template** — Ollama not running,
> timeout, malformed reply, or a failed reconciliation check. This is deliberate
> and load-bearing twice over: a missing explanation must never fail an otherwise
> valid prediction, **and** it is what will serve on free hosting, which cannot run
> a local LLM. Same code path, no branching — Ollama is simply unreachable there.
> The response carries `explanation_source: "llm" | "template"` so it is never
> ambiguous which one a reader is looking at.

*When `genre_imputed` is set*, the context genre SHAP is the **average of the
per-genre SHAP values** (legitimate — the base value is constant); phrase it as
"genre unknown, averaged across genres," not as a specific genre.

*Why SHAP, not the global "most-split features":* global importance says "energy
matters across all songs in general." SHAP says "for THIS song, high energy added
+5." The breakdown is about *this* song. Global importance is background flavor,
never the per-song claim.

**Guardrails (non-negotiable):**
- Input to the LLM is **structured facts, never raw ambiguity** — the four
  contributions, the pre-worded `*_standing` strings, a `caveats` list (never the
  raw `fame_estimated` / `genre_imputed` booleans — see the negated-boolean
  trap), and the grouped/English-labelled SHAP — never a free-form "make this a
  hit" prompt. Concept names instead of raw column names remove the temptation
  to invent acoustic detail it was never given; numbers are rounded to 2dp
  before being sent so it cannot quote more precision than it was shown.
- Every sentence traces to a number in that payload; the SHAP values must
  reconcile to the model output, or refuse to explain.
- If `caveats` is non-empty, the explanation must state it; if it is empty, the
  explanation must not invent a limitation.
- The LLM never asserts the song *will* be popular (checked, not just
  requested); it translates the composition and its drivers, and it must
  **respect the ±2–4 size of the audio part** — no dressing up a +3 as "your
  production made this a hit."
- Direction claims (above/below average, adds/subtracts) are checked against the
  actual sign in the payload, per sentence — the model does not get to overrule
  a stated fact with its own priors about what a genre "usually" does.

*Why safe to do last, why still risky:* every number already exists and was
validated by Section 3. But a wrong explanation still reads as confident and
plausible, so spot-check sentences against the SHAP numbers like unit tests.

---

## 3. Evaluation

Two numbers matter, and they must be judged separately — because one of them
looks great for the wrong reason.

### Audio model (the song part) — target is `popularity_residual`
**This is the real measure of the product.** Numbers are **low by design** — we
removed the easy variance (fame) first — so judge against a near-zero baseline.

| Level    | Spearman | R²    | Note                                   |
|----------|----------|-------|----------------------------------------|
| Minimum  | > 0.15   | > 0.05| working; clearly above baseline        |
| Good     | > 0.25   | > 0.10| audio carries real signal              |
| Strong   | > 0.35   | —     | **double-check for leakage first**     |

Current, all out-of-fold and artist-grouped:
- **Research (Spotify-10, full 66k):** residual Spearman ≈ **0.18–0.20** — above
  the floor. This is the documented finding.
- **Serving (librosa, ~9k downloaded):** ≈ **0.11** — *below* the floor. On those
  same 9k rows the Spotify features score ≈0.12, so the gap is the **sample**, not
  librosa (see Stage 6). *Measured on the 25-descriptor / first-120s extractor; the
  58-descriptor whole-track version is pending re-extraction.*

Every one of these numbers assumes **fame *and genre*** were removed first
(Stage-3 guard 3). Residualizing on fame alone reports ≈0.40 for the same model —
that is genre, not song.
*Why Spearman:* rank-based, and the percentile output is a ranking.
*Baselines:* predict-the-mean (R²=0) and a shuffled-feature control. The model
only matters if it beats these.

### Final popularity (the composed number) — target is `popularity`
`context_contribution + audio_contribution` vs actual popularity: track MAE, R²,
Spearman.

> **Honesty trap:** this R² will look *impressive* (~0.6) — but that is almost
> entirely the fame model. A high final R² does **not** mean the audio part is
> good. Never quote the composed R² as evidence the song-scoring works; that
> claim belongs only to the audio-residual metric above.

---

## 4. Standing ML principles for this repo

- **Leakage is the default failure mode.** Out-of-fold for residuals; group-split
  by artist; time-split where time matters. When a result looks too good, suspect
  leakage before celebrating.
- **Baselines before models.** A number means nothing without a dumb baseline to
  beat. (The shuffled-feature control also catches models that collapse to the
  mean and ignore their features — which happened at N=45.)
- **A control only tells you what it was built to detect.** The shuffled control
  catches leakage and mean-collapse; it is blind to residualizing on the *wrong
  confounder* (Stage-3 guard 3), which inflated our audio numbers 3.5×. Pair every
  control with a **known-value sanity check** — "the context model should score
  R²≈0.60" caught what the control could not.
- **A number's meaning depends on what you removed first.** "Audio explains X" is
  meaningless until you say what was residualized out. Same model, same data:
  0.40 with fame removed, 0.11 with fame+genre removed. Always state the baseline
  you subtracted.
- **An LLM will not reliably follow a prompt rule — enforce it, don't just ask.**
  Stage 9's three bug-fix rounds all had the same shape: giving the model
  *correct facts* (not just correct instructions) reduced errors but did not
  eliminate them — it inverted a stated "below average" fact and fabricated a
  cross-genre claim it was explicitly told not to invent. A validator that
  checks the output against the same ground truth the model was given is not
  optional polish; at a 3B model's size, it is what makes the output safe to
  ship at all.
- **Train and serve must extract features with the same code.** The serving audio
  model is only as good as its features; training and the backend both call
  `model/audio.py::extract_librosa_features`, so they can't drift silently.
- **The target is noisy.** Spotify "popularity" is a recency-weighted black box.
  Don't over-trust small differences.

---

## 5. Tech stack (MVP)
Python · pandas · scikit-learn · LightGBM · librosa (audio features) ·
**yt-dlp** (audio sourcing from YouTube) · Last.fm API (artist fame lookup) ·
SHAP (Stage 9 attribution) · **local Ollama** (Stage 9 phrasing, with a template
fallback) · joblib/JSON artifacts · **FastAPI + uvicorn** (Stage 7 serving) ·
**Next.js 15 / React 19** (Stage 8 UI).
Pinned versions in `requirements.txt`; models must load under the pinned
scikit-learn (they are pickles — see README).

## 6. Out of scope (for now)
Breakout / Top-200 chart-entry prediction (the former "Model A") · **estimating
Spotify audio features from librosa (Stage 6 "Option A") — decided against: the
librosa glossary gives explainability without the error-prone layer** ·
counterfactual "what to add to make it more popular" advice (needs a
nearest-neighbor comparison module we aren't building yet) · remix generation ·
chord-by-chord rewriting · real-time TikTok virality · "guaranteed hit" claims ·
full production React frontend.

The LLM (Stage 9) only *explains* the composed prediction — it never predicts,
and every claim must trace to a per-song SHAP value or the contribution numbers
(see Stage 9 guardrails).
