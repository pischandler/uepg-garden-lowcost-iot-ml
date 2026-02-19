from __future__ import annotations

import cv2
import numpy as np


def gray_world_rgb(rgb: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = rgb.astype(np.float32)
    means = x.reshape(-1, 3).mean(axis=0)
    g = float(means.mean())
    scale = g / (means + eps)
    y = x * scale.reshape(1, 1, 3)
    return np.clip(y, 0, 255).astype(np.uint8)


def clahe_lab_rgb(rgb: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return out.astype(np.uint8)


def gamma_rgb(rgb: np.ndarray, gamma_value: float) -> np.ndarray:
    g = float(gamma_value)
    if g <= 0:
        raise ValueError("gamma must be > 0")
    inv = 1.0 / g
    table = np.array([(i / 255.0) ** inv * 255.0 for i in range(256)], dtype=np.float32)
    out = cv2.LUT(rgb, table.astype(np.uint8))
    return out.astype(np.uint8)


def brightness_contrast_rgb(rgb: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    x = rgb.astype(np.float32)
    y = x * float(alpha) + float(beta)
    return np.clip(y, 0, 255).astype(np.uint8)


def normalize_pipeline_rgb(rgb: np.ndarray) -> np.ndarray:
    x = gray_world_rgb(rgb)
    x = clahe_lab_rgb(x, clip_limit=2.0, tile_grid=(8, 8))
    return x
