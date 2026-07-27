# SoundSignal

SoundSignal analyzes an uploaded song and estimates its Spotify-style popularity
on a scale from 0 to 100. Instead of returning only one score, it shows how the
prediction is divided between a typical-track baseline, artist fame, genre, and
the recording itself.

```text
predicted popularity = baseline + artist fame + genre + recording
```

The audio contribution is also ranked against tracks from the same genre.
Per-song SHAP values identify the strongest audio drivers, and a local Ollama
model turns the results into a plain-language explanation. If the generated
explanation contradicts the calculated result, SoundSignal replaces it with a
validated deterministic explanation.

The models were trained on 66,808 cleaned tracks and evaluated with
artist-grouped cross-validation:

| Model | MAE | RMSE | R² | Spearman |
|---|---:|---:|---:|---:|
| Artist fame + genre | 7.708 | 11.026 | 0.6198 | 0.7880 |
| Audio + artist fame + genre | **7.694** | **11.013** | **0.6206** | **0.7917** |

## Tools used

| Area | Tools |
|---|---|
| Machine learning | Python 3.14, LightGBM, scikit-learn, pandas, NumPy |
| Audio analysis | librosa |
| Explainability | SHAP, Ollama |
| Backend | FastAPI, Uvicorn |
| Frontend | Next.js 15, React 19, TypeScript |
| Artist data | Last.fm API |
| Development | Docker, Docker Compose, pytest |

## Run the application

### Configuration

Create your local environment file:

```bash
cp .env.example .env
```

Add a `LASTFM_API_KEY` to `.env` to enable live artist lookups. The application
still runs without it, but artists missing from the bundled database receive a
low estimated fame value.

### Recommended: automatic launcher

The launcher starts Ollama, downloads the configured model when necessary, and
starts the frontend and backend. It prefers Docker Compose and automatically
falls back to local Python and npm commands if Docker is unavailable.

```bash
./scripts/start.sh
```

Press `Ctrl+C` to stop everything started by the script.

### Option 1: local commands

Install the Python and frontend dependencies once:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd frontend
npm install
cd ..

ollama pull deepseek-r1:7b
```

Start Ollama:

```bash
ollama serve
```

In a second terminal, start the backend:

```bash
cd /path/to/popularity-model
source .venv/bin/activate
mkdir -p /tmp/songassess-numba-cache

NUMBA_CACHE_DIR=/tmp/songassess-numba-cache \
python -m uvicorn backend.app.main:app --reload --port 8000
```

In a third terminal, start the frontend:

```bash
cd /path/to/popularity-model/frontend
npm run dev
```

### Option 2: Docker Compose

Start Ollama and make sure the configured model is available:

```bash
ollama serve
ollama pull deepseek-r1:7b  # first run only
```

Start Docker Desktop, or use Colima from the terminal:

```bash
brew install colima  # first run only
colima start --cpus 4 --memory 6
```

Build and start the application:

```bash
docker compose up --build
```

Later starts can reuse the existing images:

```bash
docker compose up
```

Stop and remove the containers:

```bash
docker compose down
```

## Local URLs

| Service | URL |
|---|---|
| Application | http://localhost:3000 |
| API health check | http://localhost:8000/health |
| Interactive API documentation | http://localhost:8000/docs |
