#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OLLAMA_PID=""
STARTED_COLIMA=false
STARTED_COMPOSE=false
BACKEND_PID=""
FRONTEND_PID=""

log() {
  printf '\n[%s] %s\n' "SoundSignal" "$1"
}

cleanup() {
  local exit_code=$?
  trap - EXIT

  if [[ "$STARTED_COMPOSE" == true ]]; then
    log "Stopping application containers..."
    (cd "$PROJECT_DIR" && docker compose down) >/dev/null 2>&1 || true
  fi

  if [[ -n "$FRONTEND_PID" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    log "Stopping the local frontend..."
    kill "$FRONTEND_PID" 2>/dev/null || true
    wait "$FRONTEND_PID" 2>/dev/null || true
  fi

  if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    log "Stopping the local backend..."
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
  fi

  if [[ -n "$OLLAMA_PID" ]] && kill -0 "$OLLAMA_PID" 2>/dev/null; then
    log "Stopping the Ollama server started by this script..."
    kill "$OLLAMA_PID" 2>/dev/null || true
    wait "$OLLAMA_PID" 2>/dev/null || true
  fi

  if [[ "$STARTED_COLIMA" == true ]]; then
    log "Stopping Colima..."
    colima stop >/dev/null 2>&1 || true
  fi

  exit "$exit_code"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for command_name in curl ollama; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    printf 'Required command not found: %s\n' "$command_name" >&2
    exit 1
  fi
done

if [[ ! -f "$PROJECT_DIR/.env" ]]; then
  printf 'Missing %s/.env. Copy .env.example to .env and fill in the values.\n' \
    "$PROJECT_DIR" >&2
  exit 1
fi

OLLAMA_MODEL_NAME="$(
  awk -F= '
    $1 == "OLLAMA_MODEL" {
      sub(/^[^=]*=/, "")
      gsub(/\r$/, "")
      print
    }
  ' "$PROJECT_DIR/.env" | tail -n 1
)"
OLLAMA_MODEL_NAME="${OLLAMA_MODEL_NAME:-deepseek-r1:7b}"

if ! curl --fail --silent --show-error --max-time 2 \
  http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  log "Starting Ollama..."
  OLLAMA_LOG="${TMPDIR:-/tmp}/soundsignal-ollama.log"
  ollama serve >"$OLLAMA_LOG" 2>&1 &
  OLLAMA_PID=$!

  ollama_attempt=0
  until curl --fail --silent --max-time 2 \
    http://127.0.0.1:11434/api/tags >/dev/null 2>&1; do
    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
      printf 'Ollama exited before becoming ready. See %s\n' "$OLLAMA_LOG" >&2
      exit 1
    fi
    ollama_attempt=$((ollama_attempt + 1))
    if (( ollama_attempt >= 30 )); then
      printf 'Ollama did not become ready. See %s\n' "$OLLAMA_LOG" >&2
      exit 1
    fi
    sleep 1
  done
else
  log "Using the Ollama server that is already running."
fi

if ! ollama list | awk 'NR > 1 {print $1}' | grep -Fxq "$OLLAMA_MODEL_NAME"; then
  log "Downloading Ollama model $OLLAMA_MODEL_NAME..."
  ollama pull "$OLLAMA_MODEL_NAME"
else
  log "Ollama model $OLLAMA_MODEL_NAME is ready."
fi

prepare_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    log "Docker CLI is not installed; using local commands instead."
    return 1
  fi

  if docker info >/dev/null 2>&1; then
    return 0
  fi

  if command -v colima >/dev/null 2>&1; then
    if colima status >/dev/null 2>&1; then
      log "Using the Colima instance that is already running."
    else
      log "Starting Colima..."
      if ! colima start --cpus 4 --memory 6; then
        log "Colima could not start; using local commands instead."
        return 1
      fi
      STARTED_COLIMA=true
    fi
    docker context use colima >/dev/null 2>&1 || true
  elif [[ -d /Applications/Docker.app ]]; then
    log "Starting Docker Desktop..."
    if ! open -a Docker; then
      log "Docker Desktop could not start; using local commands instead."
      return 1
    fi
  else
    log "No container engine is installed; using local commands instead."
    return 1
  fi

  docker_attempt=0
  until docker info >/dev/null 2>&1; do
    docker_attempt=$((docker_attempt + 1))
    if (( docker_attempt >= 60 )); then
      log "The container engine did not become ready; using local commands instead."
      return 1
    fi
    sleep 2
  done

  if ! docker compose version >/dev/null 2>&1; then
    log "Docker Compose is unavailable; using local commands instead."
    return 1
  fi

  return 0
}

stop_started_colima() {
  if [[ "$STARTED_COLIMA" == true ]]; then
    log "Stopping Colima before switching to local commands..."
    colima stop >/dev/null 2>&1 || true
    STARTED_COLIMA=false
  fi
}

start_native() {
  local python_bin="/opt/anaconda3/envs/ml/bin/python"

  if [[ ! -x "$python_bin" ]]; then
    printf 'Pinned Python environment not found: %s\n' "$python_bin" >&2
    exit 1
  fi
  if ! command -v npm >/dev/null 2>&1; then
    printf '%s\n' 'npm is required for the local-command fallback.' >&2
    exit 1
  fi

  if [[ ! -d "$PROJECT_DIR/frontend/node_modules" ]]; then
    log "Installing frontend dependencies..."
    (cd "$PROJECT_DIR/frontend" && npm install)
  fi

  mkdir -p /tmp/songassess-numba-cache

  log "Starting the backend with local Python..."
  (
    cd "$PROJECT_DIR"
    exec env \
      ALLOWED_ORIGINS=http://localhost:3000 \
      NUMBA_CACHE_DIR=/tmp/songassess-numba-cache \
      OLLAMA_URL=http://127.0.0.1:11434 \
      "$python_bin" -m uvicorn backend.app.main:app --port 8000
  ) &
  BACKEND_PID=$!

  log "Starting the frontend with npm..."
  (
    cd "$PROJECT_DIR/frontend"
    exec npm run dev -- --hostname 0.0.0.0
  ) &
  FRONTEND_PID=$!

  native_attempt=0
  until curl --fail --silent --max-time 2 http://127.0.0.1:8000/health \
      >/dev/null 2>&1 \
    && curl --fail --silent --max-time 2 http://127.0.0.1:3000 \
      >/dev/null 2>&1; do
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
      printf '%s\n' 'The local backend exited during startup.' >&2
      return 1
    fi
    if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
      printf '%s\n' 'The local frontend exited during startup.' >&2
      return 1
    fi
    native_attempt=$((native_attempt + 1))
    if (( native_attempt >= 90 )); then
      printf '%s\n' 'The local application did not become ready within 90 seconds.' >&2
      return 1
    fi
    sleep 1
  done

  log "The local-command fallback is ready."
  printf '%s\n' \
    'Application: http://localhost:3000' \
    'API health: http://localhost:8000/health' \
    'Press Ctrl+C to stop everything.'

  while kill -0 "$BACKEND_PID" 2>/dev/null \
    && kill -0 "$FRONTEND_PID" 2>/dev/null; do
    sleep 1
  done

  printf '%s\n' 'A local application process stopped unexpectedly.' >&2
  return 1
}

if prepare_docker; then
  log "Starting the backend and frontend with Docker Compose..."
  printf '%s\n' \
    'Application: http://localhost:3000' \
    'API health: http://localhost:8000/health' \
    'Press Ctrl+C to stop everything.'

  cd "$PROJECT_DIR"
  STARTED_COMPOSE=true
  if docker compose up --build; then
    exit 0
  fi

  log "Docker Compose failed; switching to local commands."
  docker compose down >/dev/null 2>&1 || true
  STARTED_COMPOSE=false
  stop_started_colima
fi

start_native
