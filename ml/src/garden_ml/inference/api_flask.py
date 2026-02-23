from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import requests
from flask import Flask, jsonify, request
from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from garden_ml.config.logging import setup_logging
from garden_ml.config.settings import Settings
from garden_ml.data.io import fetch_bytes, secure_ext
from garden_ml.inference.predictor import load_artifacts, predict_from_image_bytes

REQ_COUNT = Counter("gml_requests_total", "Total requests", ["endpoint", "status"])
REQ_LAT = Histogram("gml_request_latency_ms", "Request latency (ms)", ["endpoint"])
FEAT_LAT = Histogram("gml_feature_latency_ms", "Feature extraction latency (ms)")
PRED_LAT = Histogram("gml_predict_latency_ms", "Prediction latency (ms)")
DECODE_LAT = Histogram("gml_decode_latency_ms", "Decode latency (ms)")


def _parse_normalize_override_query() -> bool | None:
    if "normalize" not in request.args:
        return None
    v = str(request.args.get("normalize", "")).strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _build_app(st: Settings) -> Flask:
    setup_logging(st.log_level)
    st.ensure_dirs()
    arts = load_artifacts(st.artifacts_dir)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = st.max_content_length

    @app.get("/health")
    def health():
        REQ_COUNT.labels(endpoint="/health", status="200").inc()
        return jsonify(
            ok=True,
            classes=arts.classes,
            artifacts_dir=str(st.artifacts_dir),
            photometric_normalize_default=bool(arts.photometric_normalize_default),
        )

    @app.get("/metrics")
    def metrics():
        return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

    def _infer_bytes(data: bytes, meta: dict[str, Any], normalize_override: bool | None) -> tuple[dict[str, Any], int]:
        t0 = time.perf_counter()
        try:
            cls, score, topk, timings = predict_from_image_bytes(
                arts,
                data,
                img_size=st.img_size,
                k=st.topk,
                photometric_normalize=normalize_override,
            )
            t1 = time.perf_counter()
            out = {
                "classe_predita": cls,
                "score": float(score),
                "topk": topk,
                "timings_ms": timings,
                "meta": meta,
                "photometric_normalize_used": bool(
                    arts.photometric_normalize_default if normalize_override is None else normalize_override
                ),
            }
            ep = str(meta.get("endpoint", "infer"))
            REQ_COUNT.labels(endpoint=ep, status="200").inc()
            REQ_LAT.labels(endpoint=ep).observe((t1 - t0) * 1000.0)
            if "features_ms" in timings:
                FEAT_LAT.observe(float(timings["features_ms"]))
            if "predict_ms" in timings:
                PRED_LAT.observe(float(timings["predict_ms"]))
            if "decode_ms" in timings:
                DECODE_LAT.observe(float(timings["decode_ms"]))
            return out, 200
        except Exception as e:
            t1 = time.perf_counter()
            ep = str(meta.get("endpoint", "infer"))
            REQ_COUNT.labels(endpoint=ep, status="500").inc()
            REQ_LAT.labels(endpoint=ep).observe((t1 - t0) * 1000.0)
            logger.exception("infer_failed endpoint={}", ep)
            return {"erro": "Falha durante a análise.", "mensagem": str(e)}, 500

    @app.post("/analisar")
    def analisar():
        if "image" not in request.files:
            REQ_COUNT.labels(endpoint="/analisar", status="400").inc()
            return jsonify({"erro": 'Envie o arquivo no campo "image" (multipart/form-data).'}), 400

        f = request.files["image"]
        ext = secure_ext(f.filename or "")
        if not ext:
            REQ_COUNT.labels(endpoint="/analisar", status="400").inc()
            return jsonify({"erro": "Extensão não suportada."}), 400

        data = f.read()
        meta = {
            "endpoint": "/analisar",
            "device_id": request.headers.get("X-Device-Id", ""),
        }
        normalize_override = _parse_normalize_override_query()
        out, code = _infer_bytes(data, meta, normalize_override=normalize_override)
        return jsonify(out), code

    @app.post("/analisar_url")
    def analisar_url():
        payload = request.get_json(silent=True) or {}
        url = str(payload.get("url") or "").strip()
        if not url:
            REQ_COUNT.labels(endpoint="/analisar_url", status="400").inc()
            return jsonify({"erro": "Campo obrigatório: url"}), 400

        device_id_req = str(payload.get("device_id") or "").strip()
        normalize_override = payload.get("normalize", None)
        norm = normalize_override if isinstance(normalize_override, bool) else None

        try:
            data, headers = fetch_bytes(url, timeout_s=8.0, max_bytes=int(st.max_content_length))
        except requests.RequestException as e:
            REQ_COUNT.labels(endpoint="/analisar_url", status="502").inc()
            return jsonify({"erro": "Falha ao buscar imagem.", "mensagem": str(e)}), 502
        except Exception as e:
            REQ_COUNT.labels(endpoint="/analisar_url", status="400").inc()
            return jsonify({"erro": "URL inválida ou imagem grande demais.", "mensagem": str(e)}), 400

        device_id = device_id_req or headers.get("X-Device-Id", "")

        meta = {
            "endpoint": "/analisar_url",
            "device_id": device_id,
            "source_url": url,
            "lux_raw": headers.get("X-Lux-Raw", ""),
            "soil_raw": headers.get("X-Soil-Raw", ""),
            "soil_pct": headers.get("X-Soil-Pct", ""),
            "temp_c": headers.get("X-Temp-C", ""),
            "hum_pct": headers.get("X-Hum-Pct", ""),
        }
        out, code = _infer_bytes(data, meta, normalize_override=norm)
        return jsonify(out), code

    return app


def create_app() -> Flask:
    st = Settings()
    return _build_app(st)


app = create_app()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir", type=str, default="artifacts/model_registry/v0002")
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    st = Settings(artifacts_dir=Path(args.artifacts_dir), host=str(args.host), port=int(args.port))
    a = _build_app(st)
    logger.info("serving host={} port={} artifacts={}", st.host, st.port, st.artifacts_dir)
    a.run(host=st.host, port=st.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
