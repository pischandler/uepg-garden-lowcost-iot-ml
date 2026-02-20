FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
  bash \
  make \
  build-essential \
  libgl1 \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY ml/ ./ml/

RUN pip install --no-cache-dir -U pip \
  && pip install --no-cache-dir -e "./ml[dev]" \
  && pip install --no-cache-dir gunicorn

EXPOSE 5000

CMD ["gunicorn", "-w", "1", "-k", "gthread", "--threads", "4", "-b", "0.0.0.0:5000", "garden_ml.inference.api_flask:app"]
