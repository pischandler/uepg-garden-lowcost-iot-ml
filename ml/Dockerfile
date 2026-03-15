# API image: production inference server (gunicorn).
# Build from repo root: docker build -f ml/Dockerfile .
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    bash \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY ml/ ./ml/

RUN uv pip install --system --no-cache -e "./ml"

EXPOSE 5000

LABEL org.opencontainers.image.title="garden-ml-api"
LABEL org.opencontainers.image.description="Smart Tomato Garden ML inference API"

CMD ["gunicorn", "-w", "1", "-k", "gthread", "--threads", "4", "-b", "0.0.0.0:5000", "garden_ml.inference.api_flask:app"]
