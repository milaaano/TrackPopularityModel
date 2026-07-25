---
title: SoundSignal API
emoji: 🎵
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# TrackPopularityModel

## Setup

The trained models in `model/artifacts/` are **pickles**, and pickles are tied to
the library versions that wrote them. They were produced with **scikit-learn 1.9.0
+ LightGBM 4.6.0 on Python 3.14**, and that is the only combination guaranteed to
load them correctly.

Use the conda env the project was trained in:

```bash
/opt/anaconda3/envs/ml/bin/python -m pytest tests/ -q
/opt/anaconda3/envs/ml/bin/python -m model.train        # retrain
/opt/anaconda3/envs/ml/bin/python -m model.calibrate    # rebuild the score scale
```

Or recreate it from the pins:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> **Do not use the default `python3` (anaconda base).** It has scikit-learn 1.7.2
> and no LightGBM, so loading the artifacts fails outright with
> `ModuleNotFoundError: No module named 'lightgbm'`.
>
> The *quieter* danger is a near-miss: install LightGBM into a mismatched sklearn
> and the artifacts load with only an `InconsistentVersionWarning` — a warning, not
> an error — and there is no guarantee the reconstructed estimator predicts what it
> did at training time. A silently wrong score is the one failure this project
> cannot detect from the outside, so **the serving environment must match the
> pins exactly.** If you upgrade sklearn or LightGBM, retrain and regenerate the
> artifacts rather than bumping the pin.

The backend imports `model.predictor.SongPredictor`, so whatever runs the API
(uvicorn, a worker, a container) must be started from this environment too.

## Layout

| Path | What it is |
|---|---|
| `model/features.py` | Feature lists + serving dtypes — single source of truth for training *and* inference |
| `model/train.py` | Trains the context model (fame+genre, on the 66k) and the **librosa audio model** (residual, on the downloaded set) → `model/artifacts/` |
| `model/calibrate.py` | Builds `audio_calibration.json`, the residual → 0-100 percentile scale (from the librosa audio model's out-of-fold residuals) |
| `model/predictor.py` | `SongPredictor` — the only thing the backend should import; `predict_from_audio_file(mp3)` is the upload entry point |
| `model/audio.py` | mp3 → 25 librosa descriptors — the serving audio model's inputs (Stage 6 **Option B**); the *same* extractor training uses |
| `model/fame.py` | `FameResolver` — artist name → `artists_listeners` (DB → Last.fm → low prior) |
| `notebooks/` | EDA, enrichment, `download.ipynb` (yt-dlp audio sourcing), `librosa_features.ipynb` (extraction + the Spotify-feature research finding) |
| `tests/` | Serving-path smoke tests |

## The audio model runs on librosa (Stage 6 Option B)

An uploaded mp3 can only yield **librosa** descriptors, never Spotify's engineered
features — so the serving audio model is trained directly on the 25 librosa
descriptors (`model/audio.py`), over the tracks we have downloaded and extracted
into `data/audio/` (built by `notebooks/download.ipynb` + the extraction cell of
`notebooks/librosa_features.ipynb`). The context model is unchanged — fame + genre
on the full 66k. So `python -m model.train` needs `data/audio/librosa_features.parquet`
to exist; it trains the context model on the 66k and the audio model on that
downloaded set, and errors clearly if too few tracks are present.

> The Spotify-feature audio model (residual Spearman ≈0.18–0.20 on 66k) is the
> *research* finding and lives in `notebooks/`, not the serving path.

See `CLAUDE.md` for the modeling plan and the reasoning behind the two-model
(fame vs. song) split.

## Deployment

The public application uses two services:

- **Hugging Face Docker Space** for the FastAPI API, model artifacts, and audio
  processing.
- **Vercel** for the Next.js frontend in `frontend/`.

### 1. Publish the API to a Docker Space

Create a public Hugging Face Space with **Docker** as its SDK. This deployment
branch can be pushed directly to the Space because its root `README.md`,
`Dockerfile`, and `.dockerignore` already contain the required Space
configuration:

```bash
git remote add space https://huggingface.co/spaces/<owner>/<space-name>
git push space deployment:main
```

In the Space's **Settings**, add:

- `LASTFM_API_KEY` as a **secret**.
- `ALLOWED_ORIGINS` as a **variable** after Vercel provides the production URL.

The API will be available at the Space's direct URL. Confirm it is running:

```bash
curl https://<owner>-<space-name>.hf.space/health
```

The expected response is `{"status":"ok"}`.

### 2. Publish the frontend to Vercel

Import the GitHub repository into Vercel and set:

- **Production Branch:** `deployment`
- **Root Directory:** `frontend`
- **Framework Preset:** Next.js
- **Environment Variable:**
  `NEXT_PUBLIC_API_BASE_URL=https://<owner>-<space-name>.hf.space`

Deploy the project. `NEXT_PUBLIC_API_BASE_URL` is compiled into the browser
bundle, so redeploy the frontend whenever that value changes.

### 3. Complete CORS configuration

Copy the stable Vercel production origin into the Space variable:

```text
ALLOWED_ORIGINS=https://<project>.vercel.app
```

Additional exact origins, such as a custom domain, can be comma-separated:

```text
ALLOWED_ORIGINS=https://<project>.vercel.app,https://soundsignal.example.com
```

Changing a Space variable restarts the API. After it is running again, open the
Vercel URL and test an actual audio upload. Vercel preview URLs are not allowed
by the current exact-origin CORS policy; use the production URL for the hosted
demo.

### Security note

Never commit the Last.fm API key. If a key was previously included in a
notebook or another committed file, revoke it before publishing and create a
replacement for the Space secret.
