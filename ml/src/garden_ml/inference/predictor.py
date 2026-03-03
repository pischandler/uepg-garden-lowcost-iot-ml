from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from garden_ml.config.constants import ENCODER_FILE, MODEL_FILE, TRAIN_META_FILE
from garden_ml.data.io import bgr_to_rgb, decode_bgr_from_bytes
from garden_ml.features.extract import ExtractOptions, extract_features_and_meta_from_rgb


def _load_onnx_model(onnx_path: Path) -> Any:
    """Load ONNX model with onnxruntime. Returns a wrapper with predict_proba(X) and n_features_in_."""
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    # e.g. [None, 188] -> 188
    n_features = int(next((s for s in input_shape if isinstance(s, int) and s > 0), 0))
    # probabilities: often output[1] when both label and proba exist, else output[0]
    outs = session.get_outputs()
    output_name = outs[1].name if len(outs) > 1 else outs[0].name

    class _OnnxWrapper:
        n_features_in_ = n_features

        @staticmethod
        def predict_proba(X: np.ndarray) -> np.ndarray:
            # X shape (n, 188), ensure float32 for ONNX
            feed = {input_name: X.astype(np.float32)}
            out = session.run([output_name], feed)[0]
            return np.asarray(out, dtype=np.float64)

    return _OnnxWrapper()


@dataclass(frozen=True)
class LoadedArtifacts:
    model: Any
    encoder: Any
    classes: list[str]
    img_size: int


def load_artifacts(artifacts_dir: Path) -> LoadedArtifacts:
    onnx_path = artifacts_dir / "model.onnx"
    if onnx_path.is_file():
        model = _load_onnx_model(onnx_path)
        if model is not None:
            enc = joblib.load(artifacts_dir / ENCODER_FILE)
            classes = list(enc.classes_)
            img_size_meta = 128
            meta_path = artifacts_dir / TRAIN_META_FILE
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    img_size_meta = int(meta.get("img_size", img_size_meta))
                except Exception:
                    pass
            return LoadedArtifacts(
                model=model,
                encoder=enc,
                classes=classes,
                img_size=int(img_size_meta),
            )
    model = joblib.load(artifacts_dir / MODEL_FILE)
    enc = joblib.load(artifacts_dir / ENCODER_FILE)
    classes = list(enc.classes_)

    img_size_meta = 128
    meta_path = artifacts_dir / TRAIN_META_FILE
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            img_size_meta = int(meta.get("img_size", img_size_meta))
        except Exception:
            pass

    return LoadedArtifacts(
        model=model,
        encoder=enc,
        classes=classes,
        img_size=int(img_size_meta),
    )


def predict_topk(
    arts: LoadedArtifacts,
    rgb: np.ndarray,
    k: int,
) -> tuple[str, float, list[dict[str, float]], dict[str, float], dict[str, Any]]:
    t0 = time.perf_counter()
    feat, quality = extract_features_and_meta_from_rgb(rgb, ExtractOptions(img_size=int(arts.img_size)))
    t1 = time.perf_counter()

    X = feat.reshape(1, -1).astype(np.float64)

    if hasattr(arts.model, "n_features_in_"):
        nf = int(getattr(arts.model, "n_features_in_"))
        if nf != int(X.shape[1]):
            raise RuntimeError(f"model expects {nf} features, got {int(X.shape[1])}")

    t2 = time.perf_counter()
    if not hasattr(arts.model, "predict_proba"):
        raise RuntimeError("model does not support predict_proba")
    probs = arts.model.predict_proba(X)[0]
    t3 = time.perf_counter()

    idx = int(np.argmax(probs))
    cls = str(arts.encoder.inverse_transform([idx])[0])
    score = float(probs[idx])

    top_idx = np.argsort(probs)[::-1][: int(k)]
    topk = [{"classe": str(arts.encoder.inverse_transform([int(i)])[0]), "score": float(probs[int(i)])} for i in top_idx]

    timings = {
        "features_ms": float((t1 - t0) * 1000.0),
        "predict_ms": float((t3 - t2) * 1000.0),
        "total_ms": float((t3 - t0) * 1000.0),
    }
    return cls, score, topk, timings, quality


def predict_from_image_bytes(
    arts: LoadedArtifacts,
    data: bytes,
    k: int,
    min_input_side_px: int,
) -> tuple[str, float, list[dict[str, float]], dict[str, float], dict[str, Any]]:
    t0 = time.perf_counter()
    bgr = decode_bgr_from_bytes(data)

    h, w = bgr.shape[:2]
    if int(min(h, w)) < int(min_input_side_px):
        raise ValueError(f"image too small (min side {int(min(h, w))} < {int(min_input_side_px)})")

    rgb = bgr_to_rgb(bgr)
    t1 = time.perf_counter()

    cls, score, topk, timings, quality = predict_topk(arts, rgb, k=k)
    timings["decode_ms"] = float((t1 - t0) * 1000.0)
    timings["total_ms"] = float(timings["decode_ms"] + timings["total_ms"])

    quality["input_h"] = int(h)
    quality["input_w"] = int(w)
    return cls, score, topk, timings, quality
