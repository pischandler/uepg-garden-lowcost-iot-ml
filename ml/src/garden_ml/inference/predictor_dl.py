"""MobileNet (DL) inference. Requires pip install -e '.[dl]'. Used for /predict_mobilenet and /predict_compare."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from garden_ml.config.constants import MODEL_FILE_DL
from garden_ml.data.io import bgr_to_rgb, decode_bgr_from_bytes

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from PIL import Image
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@dataclass(frozen=True)
class LoadedArtifactsDL:
    model: Any
    classes: list[str]
    img_size: int
    device: Any


def load_artifacts_dl(artifacts_dir_dl: Path) -> LoadedArtifactsDL | None:
    """Load MobileNetV3-Small checkpoint from deep_baseline output. Returns None if torch missing or dir invalid."""
    if not _TORCH_AVAILABLE:
        return None
    artifacts_dir_dl = Path(artifacts_dir_dl)
    pt_path = artifacts_dir_dl / MODEL_FILE_DL
    if not pt_path.is_file():
        return None
    try:
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model"]
        classes = list(ckpt["classes"])
        img_size = int(ckpt["img_size"])
        n_classes = len(classes)
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, n_classes)
        model.load_state_dict(state_dict, strict=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        return LoadedArtifactsDL(model=model, classes=classes, img_size=img_size, device=device)
    except Exception:
        return None


def _preprocess(rgb: np.ndarray, img_size: int) -> Any:
    """Resize and ToTensor only (matches deep_baseline tfm_test)."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch not available")
    pil = Image.fromarray(rgb)
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    x = tfm(pil)
    return x.unsqueeze(0)


def _infer(
    arts: LoadedArtifactsDL,
    tensor: Any,
    k: int,
) -> tuple[str, float, list[dict[str, float]], dict[str, float]]:
    """Run model forward; returns (cls, score, topk, timings) without decode_ms or quality."""
    t0 = time.perf_counter()
    with torch.no_grad():
        tensor = tensor.to(arts.device)
        logits = arts.model(tensor)[0]
        probs = torch.softmax(logits, dim=0).cpu().numpy()
    t1 = time.perf_counter()
    idx = int(np.argmax(probs))
    cls = arts.classes[idx]
    score = float(probs[idx])
    top_idx = np.argsort(probs)[::-1][:k]
    topk = [{"classe": arts.classes[int(i)], "score": float(probs[int(i)])} for i in top_idx]
    timings = {
        "predict_ms": float((t1 - t0) * 1000.0),
        "total_ms": float((t1 - t0) * 1000.0),
    }
    return cls, score, topk, timings


def predict_from_rgb_dl(
    arts: LoadedArtifactsDL,
    rgb: np.ndarray,
    k: int,
) -> tuple[str, float, list[dict[str, float]], dict[str, float], dict[str, Any]]:
    """Predict from RGB array (for /predict_compare when image already decoded). Same return shape as classical."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch not available")
    t0 = time.perf_counter()
    tensor = _preprocess(rgb, arts.img_size)
    t1 = time.perf_counter()
    cls, score, topk, timings = _infer(arts, tensor, k)
    timings["preprocess_ms"] = float((t1 - t0) * 1000.0)
    timings["total_ms"] = float(timings["total_ms"]) + timings["preprocess_ms"]
    h, w = rgb.shape[:2]
    quality = {"input_h": int(h), "input_w": int(w)}
    return cls, score, topk, timings, quality


def predict_from_image_bytes_dl(
    arts: LoadedArtifactsDL,
    data: bytes,
    k: int,
    min_input_side_px: int,
) -> tuple[str, float, list[dict[str, float]], dict[str, float], dict[str, Any]]:
    """Decode JPEG, then predict. Same return shape as predictor.predict_from_image_bytes."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch not available")
    t0 = time.perf_counter()
    bgr = decode_bgr_from_bytes(data)
    h, w = bgr.shape[:2]
    if int(min(h, w)) < int(min_input_side_px):
        raise ValueError(f"image too small (min side {int(min(h, w))} < {int(min_input_side_px)})")
    rgb = bgr_to_rgb(bgr)
    t1 = time.perf_counter()
    decode_ms = float((t1 - t0) * 1000.0)
    cls, score, topk, timings, quality = predict_from_rgb_dl(arts, rgb, k)
    timings["decode_ms"] = decode_ms
    timings["total_ms"] = float(timings["total_ms"]) + decode_ms
    quality["input_h"] = int(h)
    quality["input_w"] = int(w)
    return cls, score, topk, timings, quality
