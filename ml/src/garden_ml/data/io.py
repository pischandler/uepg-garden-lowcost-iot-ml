from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from PIL import Image

from garden_ml.config.constants import ALLOWED_IMAGE_EXT


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_IMAGE_EXT


def read_rgb_pil(path: Path) -> np.ndarray:
    pil = Image.open(path).convert("RGB")
    return np.asarray(pil, dtype=np.uint8)


def decode_bgr_from_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("invalid image bytes")
    return bgr


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def fetch_bytes(url: str, timeout_s: float = 8.0, max_bytes: int = 8 * 1024 * 1024) -> tuple[bytes, dict[str, str]]:
    u = urlparse(url)
    if u.scheme not in {"http", "https"}:
        raise ValueError("only http/https urls are allowed")

    r = requests.get(url, timeout=timeout_s, stream=True)
    r.raise_for_status()

    buf = bytearray()
    for chunk in r.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        buf.extend(chunk)
        if len(buf) > int(max_bytes):
            raise ValueError("response too large")

    headers = {k: v for k, v in r.headers.items()}
    return bytes(buf), headers


def secure_ext(filename: str) -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    return ext if ext in ALLOWED_IMAGE_EXT else ""
