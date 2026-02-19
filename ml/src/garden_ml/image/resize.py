from __future__ import annotations

import cv2
import numpy as np


def letterbox_rgb(rgb: np.ndarray, size: tuple[int, int], bg: int = 0) -> np.ndarray:
    h, w = rgb.shape[:2]
    th, tw = size
    scale = min(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((th, tw, 3), bg, dtype=np.uint8)
    top = (th - nh) // 2
    left = (tw - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas
