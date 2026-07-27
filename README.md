---
title: SoundSignal API
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# SoundSignal — song popularity, decomposed

Upload a track, name the artist, pick a genre. SoundSignal predicts a
Spotify-style popularity score from 0 to 100 **and breaks that number down into
what produced it** — because a single blended score hides the thing a musician
most wants to know: how much of this is my reputation, and how much is the record
I just made?

```
predicted popularity  =  typical track  +  artist fame  +  genre  +  this recording
        74.5          =      38.7       +      +3.4     +  +30.3  +      +2.1
                             \__ what the artist brings __/   \_ what the song adds _/
```

The four terms sum to the prediction exactly — that identity is asserted in the
API and pinned by tests, so the breakdown is real arithmetic rather than a
plausible-looking gloss. The last term also comes with a **within-genre
percentile** ("better than 76% of hip-hop tracks"), because +2.1 on its own
doesn't tell you whether +2.1 is good, and judging a pop track's audio against a
classical track's would not be a fair comparison.

## Performance

Trained on **66,808 cleaned tracks** (from a raw 114,000), evaluated with a
10-fold split grouped by artist so no artist appears in both train and test:

| Model | MAE | RMSE | R² | Spearman |
|---|---|---|---|---|
| Fame + genre only | 7.708 | 11.026 | 0.6198 | 0.7880 |
| Audio + fame + genre | **7.694** | **11.013** | **0.6206** | **0.7917** |

Artist fame and genre explain most of the variance, so the term attributable to
the recording itself is small by design — a few points. That is a measured
finding rather than a weak model, and the interface states it instead of
dressing it up. Full reasoning, including why the audio model runs on `librosa`
descriptors and why removing genre inflates the apparent audio signal ~3.5×, is
in [`report/report.pdf`](report/report.pdf).

## Quick start

```bash
./scripts/start.sh
```

Starts Ollama and pulls the configured model if needed, then brings up the app —
preferring Docker Compose, falling back to running the backend and frontend
directly if Docker is unavailable. `Ctrl+C` stops everything it started.

| | |
|---|---|
| Application | http://localhost:3000 |
| API health | http://localhost:8000/health |
| API docs | http://localhost:8000/docs |

Set `LASTFM_API_KEY` in the root `.env` to enable live artist lookups.

<details>
<summary>Run the two services manually</summary>

```bash
# terminal 1 — backend
mkdir -p /tmp/songassess-numba-cache
NUMBA_CACHE_DIR=/tmp/songassess-numba-cache \
  python -m uvicorn backend.app.main:app --reload --port 8000

# terminal 2 — frontend
cd frontend && npm install && npm run dev
```
</details>

Tests: `python -m pytest tests/ -q`

## Technologies

| Area | Stack |
|---|---|
| Modelling | Python, LightGBM, scikit-learn, pandas, NumPy |
| Audio | librosa (58 descriptors from raw mp3), yt-dlp (corpus collection) |
| Explanation | SHAP (per-song attribution), Ollama (local LLM narration) |
| Backend | FastAPI, uvicorn |
| Frontend | Next.js 15, React 19, TypeScript |
| Data | Spotify tracks dataset, Last.fm API (artist listeners) |
| Packaging | Docker, Docker Compose, pytest |

## Project layout

| Path | What it is |
|---|---|
| `model/features.py` | Feature lists and serving dtypes — one source of truth for training and inference |
| `model/train.py` | Trains the context model (fame + genre) and the audio model → `model/artifacts/` |
| `model/calibrate.py` | Builds the residual → 0–100 percentile scale, per genre |
| `model/predictor.py` | `SongPredictor` — the only module the backend imports; `predict_from_audio_file(mp3)` is the upload entry point |
| `model/audio.py` | mp3 → 58 librosa descriptors; the same extractor used in training and serving |
| `model/fame.py` | `FameResolver` — artist name → listener count (local DB → Last.fm → low prior) |
| `model/explain.py` | Grouped SHAP values and the plain-English explanation layer |
| `backend/app/main.py` | FastAPI routes |
| `frontend/` | Next.js single-page UI |
| `notebooks/` | EDA, dataset enrichment, audio download and feature extraction |
| `tests/` | Serving-path tests |

## Environment

The files in `model/artifacts/` are pickles, and pickles are tied to the library
versions that wrote them: **scikit-learn 1.9.0, LightGBM 4.6.0, Python 3.14**. A
mismatched version may still load them, emitting only a warning, and then predict
differently — a silently wrong score is the one failure this project cannot
detect from the outside. Install from `requirements.txt` exactly, and if you
upgrade scikit-learn or LightGBM, retrain and regenerate the artifacts rather
than bumping the pin.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Deployment

The hosted demo runs as two services: a **Hugging Face Docker Space** for the API
(this repo's root `Dockerfile` and the frontmatter above configure it) and
**Vercel** for the frontend in `frontend/`.

```bash
git remote add space https://huggingface.co/spaces/<owner>/<space-name>
git push space main
```

In the Space settings add `LASTFM_API_KEY` as a **secret** and `ALLOWED_ORIGINS`
as a **variable**. On Vercel, set the root directory to `frontend` and
`NEXT_PUBLIC_API_BASE_URL` to the Space URL. Two things that are easy to miss:

- `NEXT_PUBLIC_API_BASE_URL` is compiled into the browser bundle, so the frontend
  must be redeployed whenever it changes.
- CORS matches exact origins, so Vercel preview URLs are rejected — use the
  production URL.

Never commit the Last.fm key; if one was ever committed, revoke it and issue a
replacement for the Space secret.
