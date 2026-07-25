FROM python:3.14-slim

# Audio decoding for librosa plus the OpenMP runtime used by LightGBM.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgomp1 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Docker Spaces run containers as UID 1000.
RUN useradd --create-home --uid 1000 user

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    NUMBA_CACHE_DIR=/tmp/numba-cache \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements-serve.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-serve.txt

# Training data and notebooks are intentionally not part of the serving image.
COPY --chown=user backend ./backend
COPY --chown=user model ./model

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
