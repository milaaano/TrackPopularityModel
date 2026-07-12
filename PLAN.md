# SongAssess July 17 MVP + Free Deployment Plan

## Summary
- Target: a recruiter-facing public MVP by **Friday, July 17, 2026**.
- Product flow: a React/Next.js page lets a user upload an audio file, enter artist name, genre, and explicit/non-explicit status, then shows popularity prediction, audio adjustment, SHAP drivers, and a grounded explanation.
- Architecture: **React/Next.js frontend only** + **FastAPI backend**. Next.js should not handle ML/audio backend work.
- Free deployment target: **Vercel Hobby for the frontend** and **Hugging Face Docker Space for the FastAPI backend**.

## Core Implementation
- Convert notebook work into a reusable Python package:
  - `src/songassess/models/train.py`: trains models and saves artifacts.
  - `src/songassess/models/predictor.py`: loads artifacts and predicts for FastAPI.
  - `src/songassess/audio/extract.py`: uploaded audio file to audio descriptors.
  - `src/songassess/audio/spotify_proxy.py`: raw audio descriptors to estimated Spotify-style features.
  - `backend/app/main.py`: FastAPI routes only.
- Train two main LightGBM models:
  - Context model: `artist/context features -> expected popularity`.
  - Audio residual model: `audio features -> popularity leftover after context`.
- Use `data/processed/orig_data.parquet` for training because it includes `primary_artist`, which is needed for group-aware splitting.
- Use `librosa` as the mandatory July 17 audio extractor. Essentia is optional only if install/import/sample extraction works quickly. AudioFlux is out for v1.

## Model Training Design
- Split data with artist groups:
  - First create a final untouched test set using `GroupShuffleSplit` by `primary_artist`.
  - On the training set, create out-of-fold context predictions with `GroupKFold`, preferably 10 folds.
- Compute the residual target:
  - `popularity_residual = actual_popularity - out_of_fold_context_prediction`.
  - This residual is the **target** for the audio model, not an input feature.
- Final training flow:
  - Train fold-specific context models only to generate leakage-safe residuals.
  - Train the audio residual model on audio features and `popularity_residual`.
  - Train one final context model on the full training set for inference.
  - Save a backend-ready artifact bundle with `joblib`.
- At inference:
  - `context_prediction = final_context_model(context_features)`.
  - `audio_adjustment = audio_residual_model(audio_features)`.
  - `final_prediction = clip(context_prediction + audio_adjustment, 0, 100)`.
- Keep a full-feature model only as a benchmark, not as the main product explanation.

## Spotify-Style Feature Reproduction
- MVP target: download audio for **2,000-3,000 songs** from the existing dataset to train the Spotify-style feature estimator.
- Minimum useful smoke test: about **500 songs**.
- Better target if time allows: **5,000+ songs**, especially for harder targets like `valence`, `acousticness`, `instrumentalness`, and `speechiness`.
- Sampling should be stratified:
  - Cover many genres.
  - Cover low/medium/high ranges of Spotify audio features.
  - Avoid duplicates and near-duplicates.
  - Keep a clean holdout set, ideally grouped by artist.
- If only 30-second previews are available, label outputs as preview-based approximations, not exact Spotify audio features.

## API And Frontend Contract
- FastAPI owns the backend:
  - `POST /analyze` accepts multipart form data: `audio_file`, `artist_name`, `genre`, `explicit`.
  - `GET /health` supports cold-start wake-up and deployment checks.
- `/analyze` response includes:
  - `context_popularity`
  - `audio_adjustment`
  - `final_popularity`
  - `raw_audio_features`
  - `estimated_audio_features`
  - `shap_context`
  - `shap_audio`
  - `confidence`
  - `warnings`
  - `explanation`
- React/Next.js frontend:
  - Calls FastAPI directly with `fetch(apiBaseUrl + "/analyze")`.
  - Uploads audio directly to FastAPI, not through Next.js API routes.
  - Calls `/health` on page load to wake the backend while the recruiter reads the page.

## Deployment
- Frontend:
  - Deploy `web/` to Vercel Hobby.
  - Set `NEXT_PUBLIC_API_BASE_URL=https://<your-space>.hf.space`.
  - The page itself should load in under 2-3 seconds.
- Backend:
  - Deploy FastAPI as a Hugging Face Docker Space on free CPU hardware.
  - Expose `/health` and `/analyze` on port `7860`.
  - Package trained model artifacts with the Space or download them during build from a public artifact location.
  - Configure CORS for the Vercel production URL.
- Hosted explanation:
  - Use deterministic template explanation online, grounded in model outputs and SHAP values.
  - Keep Ollama as local/full-demo mode only; hosted Ollama is out of scope for free deployment.
- Cold-start UX:
  - First analysis after backend sleep may take **30-120 seconds**.
  - Warm analysis should target **10-30 seconds**, depending on audio length and SHAP cost.
  - Show a clear "Waking analysis service..." state and a friendly retry message after a long timeout.

## Schedule
- **July 9:** freeze MVP scope, schemas, repo structure, free deployment target, and artifact strategy.
- **July 10:** create Python 3.11 env, install deps, smoke-test `librosa`, optional Essentia, FastAPI, SHAP, model library, Docker, and local Ollama.
- **July 11:** productionize training: group-aware split, out-of-fold context predictions, residual target, audio residual model, final context model, artifact saving.
- **July 12:** implement audio extraction and Spotify-like proxy estimator; start with 500-song smoke test, then expand toward 2,000-3,000 songs.
- **July 13:** build FastAPI `/analyze`, `/health`, model loading, file validation, temp-file cleanup, and response schema.
- **July 14:** build one-page React/Next.js UI: dropzone, artist input, genre selector, explicit toggle, analyze state, wake-up state, result panels.
- **July 15:** add SHAP outputs, template explanation fallback, CORS config, Dockerfile, upload size limit, and frontend backend wake-up flow.
- **July 16:** deploy backend to Hugging Face Spaces and frontend to Vercel; test cold start, upload, prediction, SHAP display, and error states.
- **July 17:** final recruiter demo pass, README deployment notes, GitHub link, public Vercel URL, limitations, and screenshots.

## Test Plan
- Model tests: reproducible train command, group-aware split sanity check, feature schema checks, artifact load test, output clamped to 0-100, context-only baseline, zero-residual baseline, and full-feature benchmark comparison.
- Audio tests: valid MP3/WAV/M4A upload, short audio, corrupt audio, oversized file, unsupported file type, and preview-length warning.
- API tests: valid upload, missing file, unknown artist, unknown genre, invalid explicit value, `/health`, CORS, and temp-file cleanup.
- UI tests: page loads fast, backend wake-up state appears, form validation works, analyze button disables while running, errors are readable, and result panels render on desktop/mobile.
- Hosted demo tests: Vercel page loads, Hugging Face backend wakes, upload completes, result JSON renders, fallback explanation appears, and retry messaging works.

## Assumptions And References
- Assumption: July 17 means a public recruiter demo URL, not production-grade uptime.
- Assumption: free hosting is preferred over always-on reliability, so cold starts are acceptable if handled cleanly.
- Assumption: no live Spotify API dependency; Spotify restricted new access to Audio Features/Audio Analysis in its Nov. 27, 2024 API changes.
- Assumption: `artist_fame_loo` is acceptable for MVP, but should be recomputed carefully or replaced with external artist metadata later because it is derived from popularity.
- References: [Spotify API changes](https://developer.spotify.com/blog/2024-11-27-changes-to-the-web-api), [librosa feature docs](https://librosa.org/doc/latest/feature.html), [Essentia docs](https://essentia.upf.edu/documentation.html), [Vercel Hobby Plan](https://vercel.com/docs/plans/hobby), [Hugging Face Spaces Overview](https://huggingface.co/docs/hub/spaces-overview), [Hugging Face Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker), [Render Free Limits](https://render.com/docs/free), [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md).
