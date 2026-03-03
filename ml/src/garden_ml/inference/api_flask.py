from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import orjson
import requests
from flask import Flask, request
from flask import Response as FlaskResponse
from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from garden_ml.config.logging import setup_logging
from garden_ml.config.settings import Settings
from garden_ml.data.io import bgr_to_rgb, decode_bgr_from_bytes, fetch_bytes, secure_ext
from garden_ml.inference.serializers import PredictionResponse, PredictionTopK
from garden_ml.inference.storage import DebugStorage
from garden_ml.inference.predictor import load_artifacts, predict_from_image_bytes, predict_topk
from garden_ml.inference.predictor_dl import load_artifacts_dl, predict_from_image_bytes_dl, predict_from_rgb_dl

REQ_COUNT = Counter("gml_requests_total", "Total requests", ["endpoint", "status"])
REQ_LAT = Histogram("gml_request_latency_ms", "Request latency (ms)", ["endpoint"])
FEAT_LAT = Histogram("gml_feature_latency_ms", "Feature extraction latency (ms)")
PRED_LAT = Histogram("gml_predict_latency_ms", "Prediction latency (ms)")
DECODE_LAT = Histogram("gml_decode_latency_ms", "Decode latency (ms)")


def _json_response(obj: Any, status: int = 200) -> FlaskResponse:
    """JSON response using orjson for faster serialization."""
    return FlaskResponse(
        orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY),
        status=status,
        mimetype="application/json",
    )


def _build_prediction_response(
    cls: str,
    score: float,
    topk: list[dict[str, Any]],
    timings: dict[str, float],
    quality: dict[str, Any],
    meta: dict[str, Any],
    min_conf: float,
    min_mask: float,
    min_mean_v: float,
    min_lap: float,
    model_img_size: int,
) -> PredictionResponse:
    """Build PredictionResponse from raw prediction and thresholds (shared by classical and DL)."""
    reasons: list[str] = []
    mask_cov = float(quality.get("mask_coverage", 1.0))
    if min_mask > 0 and mask_cov < min_mask:
        reasons.append("low_mask_coverage")
    if float(score) < min_conf:
        reasons.append("low_confidence")
    mean_v = quality.get("mean_v", None)
    if mean_v is not None and min_mean_v > 0 and float(mean_v) < min_mean_v:
        reasons.append("low_light")
    lap_var = quality.get("laplacian_var", None)
    if lap_var is not None and min_lap > 0 and float(lap_var) < min_lap:
        reasons.append("blurry")
    confident = len(reasons) == 0
    pred_cls: str | None = cls if confident else None
    return PredictionResponse(
        classe_predita=pred_cls,
        score=float(score),
        topk=[PredictionTopK(**x) for x in topk],
        timings_ms={k: float(v) for k, v in timings.items()},
        quality=quality,
        meta=meta,
        confident=bool(confident),
        reasons=reasons,
        min_confidence=min_conf,
        min_mask_coverage=min_mask,
        min_mean_v=min_mean_v,
        min_laplacian_var=min_lap,
        model_img_size=model_img_size,
    )


def _build_app(st: Settings) -> Flask:
    setup_logging(st.log_level)
    st.ensure_dirs()
    arts = load_artifacts(st.artifacts_dir)
    arts_dl = None
    if st.artifacts_dir_dl is not None and st.artifacts_dir_dl.is_dir():
        arts_dl = load_artifacts_dl(st.artifacts_dir_dl)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = st.max_content_length

    debug_store: DebugStorage | None = None
    if bool(getattr(st, "save_debug", False)):
        debug_store = DebugStorage(root=st.debug_root / st.artifacts_dir.name)
        debug_store.ensure()
        debug_store.ensure_csv()

    def _fget(name: str, default: float = 0.0) -> float:
        """pega float do Settings se existir"""
        v = getattr(st, name, default)
        try:
            return float(v)
        except Exception:
            return float(default)

    @app.get("/health")
    def health():
        REQ_COUNT.labels(endpoint="/health", status="200").inc()
        payload = {
            "ok": True,
            "classes": arts.classes,
            "artifacts_dir": str(st.artifacts_dir),
            "model_img_size": int(arts.img_size),
            "min_input_side_px": int(st.min_input_side_px),
            "min_confidence": float(st.min_confidence),
            "min_mask_coverage": float(getattr(st, "min_mask_coverage", 0.0)),
            "min_mean_v": _fget("min_mean_v", 0.0),
            "min_laplacian_var": _fget("min_laplacian_var", 0.0),
            "backends": {"classical": True, "mobilenet": arts_dl is not None},
        }
        if arts_dl is not None:
            payload["artifacts_dir_dl"] = str(st.artifacts_dir_dl)
        return _json_response(payload)

    @app.get("/metrics")
    def metrics():
        return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

    def _infer_bytes(data: bytes, meta: dict[str, Any]) -> tuple[dict[str, Any], int]:
        t0 = time.perf_counter()
        try:
            cls, score, topk, timings, quality = predict_from_image_bytes(
                arts,
                data,
                k=st.topk,
                min_input_side_px=int(st.min_input_side_px),
            )

            min_conf = float(st.min_confidence)
            min_mask = float(getattr(st, "min_mask_coverage", 0.0))
            min_mean_v = _fget("min_mean_v", 0.0)
            min_lap = _fget("min_laplacian_var", 0.0)
            resp = _build_prediction_response(
                cls, score, topk, timings, quality, meta,
                min_conf, min_mask, min_mean_v, min_lap, int(arts.img_size),
            )

            t1 = time.perf_counter()
            ep = str(meta.get("endpoint", "infer"))
            REQ_COUNT.labels(endpoint=ep, status="200").inc()
            REQ_LAT.labels(endpoint=ep).observe((t1 - t0) * 1000.0)

            if "features_ms" in timings:
                FEAT_LAT.observe(float(timings["features_ms"]))
            if "predict_ms" in timings:
                PRED_LAT.observe(float(timings["predict_ms"]))
            if "decode_ms" in timings:
                DECODE_LAT.observe(float(timings["decode_ms"]))

            # debug: salva JPEG bruto + imagem decodificada
            if debug_store is not None:
                try:
                    device_id = str(meta.get("device_id", "") or "")
                    base = str(int(time.time() * 1000.0))

                    # salva o JPEG bruto (ótimo p/ auditoria)
                    raw_dir = debug_store.root / "raw_jpeg"
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    (raw_dir / f"{base}.jpg").write_bytes(data)

                    # salva BGR original (já existia)
                    bgr = decode_bgr_from_bytes(data)
                    debug_store.save_original_bgr(bgr, base=base)

                    debug_store.append_row(
                        device_id=device_id,
                        source=str(meta.get("endpoint", "")),
                        classe=str(resp.classe_predita or ""),
                        score=float(resp.score),
                        total_ms=float(resp.timings_ms.get("total_ms", 0.0)),
                        lux_raw=str(meta.get("lux_raw", "")),
                        soil_raw=str(meta.get("soil_raw", "")),
                        temp_c=str(meta.get("temp_c", "")),
                        hum_pct=str(meta.get("hum_pct", "")),
                    )
                except Exception:
                    logger.exception("debug_storage_failed")

            return _json_response(resp.model_dump(), 200)

        except Exception as e:
            t1 = time.perf_counter()
            ep = str(meta.get("endpoint", "infer"))
            REQ_COUNT.labels(endpoint=ep, status="500").inc()
            REQ_LAT.labels(endpoint=ep).observe((t1 - t0) * 1000.0)
            logger.exception("infer_failed endpoint={}", ep)
            return _json_response({"erro": "Falha durante a análise.", "mensagem": str(e)}, 500)

    # ✅ endpoint para o ESP (JPEG bruto no body)
    @app.post("/predict")
    def predict_raw():
        data = request.get_data(cache=False) or b""
        if not data:
            REQ_COUNT.labels(endpoint="/predict", status="400").inc()
            return _json_response({"erro": "Envie bytes JPEG no body (Content-Type: image/jpeg)."}, 400)

        meta = {
            "endpoint": "/predict",
            "device_id": request.headers.get("X-Device-Id", ""),
            "lux_raw": request.headers.get("X-Lux-Raw", ""),
            "soil_raw": request.headers.get("X-Soil-Raw", ""),
            "soil_pct": request.headers.get("X-Soil-Pct", ""),
            "temp_c": request.headers.get("X-Temp-C", ""),
            "hum_pct": request.headers.get("X-Hum-Pct", ""),
            "pump_on": request.headers.get("X-Pump-On", ""),
        }
        out, code = _infer_bytes(data, meta)
        return _json_response(out, code)

    @app.post("/predict_mobilenet")
    def predict_mobilenet():
        if arts_dl is None:
            REQ_COUNT.labels(endpoint="/predict_mobilenet", status="503").inc()
            return _json_response({
                "erro": "Modelo MobileNet não disponível.",
                "mensagem": "Configure artifacts_dir_dl e instale extras: pip install -e '.[dl]'",
            }, 503)
        data = request.get_data(cache=False) or b""
        if not data:
            REQ_COUNT.labels(endpoint="/predict_mobilenet", status="400").inc()
            return _json_response({"erro": "Envie bytes JPEG no body (Content-Type: image/jpeg)."}, 400)
        meta = {
            "endpoint": "/predict_mobilenet",
            "device_id": request.headers.get("X-Device-Id", ""),
        }
        t0 = time.perf_counter()
        try:
            cls, score, topk, timings, quality = predict_from_image_bytes_dl(
                arts_dl, data, k=st.topk, min_input_side_px=int(st.min_input_side_px)
            )
            min_conf = float(st.min_confidence)
            min_mask = float(getattr(st, "min_mask_coverage", 0.0))
            min_mean_v = _fget("min_mean_v", 0.0)
            min_lap = _fget("min_laplacian_var", 0.0)
            resp = _build_prediction_response(
                cls, score, topk, timings, quality, meta,
                min_conf, min_mask, min_mean_v, min_lap, int(arts_dl.img_size),
            )
            t1 = time.perf_counter()
            REQ_COUNT.labels(endpoint="/predict_mobilenet", status="200").inc()
            REQ_LAT.labels(endpoint="/predict_mobilenet").observe((t1 - t0) * 1000.0)
            return _json_response(resp.model_dump(), 200)
        except Exception as e:
            t1 = time.perf_counter()
            REQ_COUNT.labels(endpoint="/predict_mobilenet", status="500").inc()
            REQ_LAT.labels(endpoint="/predict_mobilenet").observe((t1 - t0) * 1000.0)
            logger.exception("predict_mobilenet_failed")
            return _json_response({"erro": "Falha durante a análise (MobileNet).", "mensagem": str(e)}, 500)

    @app.post("/predict_compare")
    def predict_compare():
        if arts_dl is None:
            REQ_COUNT.labels(endpoint="/predict_compare", status="503").inc()
            return _json_response({
                "erro": "Comparativo indisponível: modelo MobileNet não carregado.",
                "mensagem": "Configure artifacts_dir_dl e instale pip install -e '.[dl]'",
            }, 503)
        data = request.get_data(cache=False) or b""
        if not data:
            REQ_COUNT.labels(endpoint="/predict_compare", status="400").inc()
            return _json_response({"erro": "Envie bytes JPEG no body (Content-Type: image/jpeg)."}, 400)
        meta = {"endpoint": "/predict_compare", "device_id": request.headers.get("X-Device-Id", "")}
        t0 = time.perf_counter()
        try:
            bgr = decode_bgr_from_bytes(data)
            h, w = bgr.shape[:2]
            if int(min(h, w)) < int(st.min_input_side_px):
                REQ_COUNT.labels(endpoint="/predict_compare", status="400").inc()
                return _json_response({
                    "erro": f"Imagem pequena (lado mínimo {int(min(h, w))} < {int(st.min_input_side_px)}).",
                }, 400)
            rgb = bgr_to_rgb(bgr)
            t_decode = time.perf_counter()
            decode_ms = float((t_decode - t0) * 1000.0)

            cls_c, score_c, topk_c, timings_c, quality_c = predict_topk(arts, rgb, k=st.topk)
            timings_c["decode_ms"] = decode_ms
            timings_c["total_ms"] = float(timings_c.get("total_ms", 0)) + decode_ms
            quality_c["input_h"] = int(h)
            quality_c["input_w"] = int(w)

            cls_d, score_d, topk_d, timings_d, quality_d = predict_from_rgb_dl(arts_dl, rgb, k=st.topk)
            timings_d["decode_ms"] = decode_ms
            timings_d["total_ms"] = float(timings_d.get("total_ms", 0)) + decode_ms
            quality_d["input_h"] = int(h)
            quality_d["input_w"] = int(w)

            min_conf = float(st.min_confidence)
            min_mask = float(getattr(st, "min_mask_coverage", 0.0))
            min_mean_v = _fget("min_mean_v", 0.0)
            min_lap = _fget("min_laplacian_var", 0.0)
            resp_c = _build_prediction_response(
                cls_c, score_c, topk_c, timings_c, quality_c, meta,
                min_conf, min_mask, min_mean_v, min_lap, int(arts.img_size),
            )
            resp_d = _build_prediction_response(
                cls_d, score_d, topk_d, timings_d, quality_d, meta,
                min_conf, min_mask, min_mean_v, min_lap, int(arts_dl.img_size),
            )
            t1 = time.perf_counter()
            REQ_COUNT.labels(endpoint="/predict_compare", status="200").inc()
            REQ_LAT.labels(endpoint="/predict_compare").observe((t1 - t0) * 1000.0)
            return _json_response({
                "classical": resp_c.model_dump(),
                "mobilenet": resp_d.model_dump(),
            }, 200)
        except Exception as e:
            t1 = time.perf_counter()
            REQ_COUNT.labels(endpoint="/predict_compare", status="500").inc()
            REQ_LAT.labels(endpoint="/predict_compare").observe((t1 - t0) * 1000.0)
            logger.exception("predict_compare_failed")
            return _json_response({"erro": "Falha no comparativo.", "mensagem": str(e)}, 500)

    @app.post("/analisar")
    def analisar():
        if "normalize" in request.args:
            REQ_COUNT.labels(endpoint="/analisar", status="400").inc()
            return _json_response({"erro": "Parâmetro normalize não é suportado no v0004."}, 400)

        if "image" not in request.files:
            REQ_COUNT.labels(endpoint="/analisar", status="400").inc()
            return _json_response({"erro": 'Envie o arquivo no campo "image" (multipart/form-data).'}, 400)

        f = request.files["image"]
        ext = secure_ext(f.filename or "")
        if not ext:
            REQ_COUNT.labels(endpoint="/analisar", status="400").inc()
            return _json_response({"erro": "Extensão não suportada."}, 400)

        data = f.read()
        meta = {"endpoint": "/analisar", "device_id": request.headers.get("X-Device-Id", "")}
        out, code = _infer_bytes(data, meta)
        return _json_response(out, code)

    @app.post("/analisar_url")
    def analisar_url():
        payload = request.get_json(silent=True) or {}
        if "normalize" in payload:
            REQ_COUNT.labels(endpoint="/analisar_url", status="400").inc()
            return _json_response({"erro": "Campo normalize não é suportado no v0004."}, 400)

        url = str(payload.get("url") or "").strip()
        if not url:
            REQ_COUNT.labels(endpoint="/analisar_url", status="400").inc()
            return _json_response({"erro": "Campo obrigatório: url"}, 400)

        device_id_req = str(payload.get("device_id") or "").strip()

        try:
            data, headers = fetch_bytes(url, timeout_s=8.0, max_bytes=int(st.max_content_length))
        except requests.RequestException as e:
            REQ_COUNT.labels(endpoint="/analisar_url", status="502").inc()
            return _json_response({"erro": "Falha ao buscar imagem.", "mensagem": str(e)}, 502)
        except Exception as e:
            REQ_COUNT.labels(endpoint="/analisar_url", status="400").inc()
            return _json_response({"erro": "URL inválida ou imagem grande demais.", "mensagem": str(e)}, 400)

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
        out, code = _infer_bytes(data, meta)
        return _json_response(out, code)

    return app


def create_app() -> Flask:
    st = Settings()
    return _build_app(st)


app = create_app()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir", type=str, default="artifacts/model_registry/v0004")
    ap.add_argument("--artifacts_dir_dl", type=str, default=None, help="MobileNet artifacts dir (enables /predict_mobilenet and /predict_compare)")
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    st_kw: dict = {"artifacts_dir": Path(args.artifacts_dir), "host": str(args.host), "port": int(args.port)}
    if args.artifacts_dir_dl is not None:
        st_kw["artifacts_dir_dl"] = Path(args.artifacts_dir_dl)
    st = Settings(**st_kw)
    a = _build_app(st)
    logger.info("serving host={} port={} artifacts={}", st.host, st.port, st.artifacts_dir)
    a.run(host=st.host, port=st.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
