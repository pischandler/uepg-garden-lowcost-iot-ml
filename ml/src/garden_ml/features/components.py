from __future__ import annotations

import cv2
import mahotas
import numpy as np


def haralick_13(gray: np.ndarray) -> np.ndarray:
    return mahotas.features.haralick(gray).mean(axis=0).astype(np.float64)


def zernike_25(mask: np.ndarray) -> np.ndarray:
    radius = min(mask.shape[0], mask.shape[1]) // 2
    z = mahotas.features.zernike_moments((mask > 0).astype(np.uint8), radius, degree=8).astype(np.float64)
    if z.size != 25:
        raise ValueError(f"zernike size {z.size} != 25")
    return z


def hsv_hist_48(segmented_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2HSV)
    m = (mask > 0).astype(np.uint8)

    h_hist = cv2.calcHist([hsv], [0], m, [16], [0, 180]).flatten().astype(np.float64)
    s_hist = cv2.calcHist([hsv], [1], m, [16], [0, 256]).flatten().astype(np.float64)
    v_hist = cv2.calcHist([hsv], [2], m, [16], [0, 256]).flatten().astype(np.float64)

    h_hist /= (h_hist.sum() + 1e-12)
    s_hist /= (s_hist.sum() + 1e-12)
    v_hist /= (v_hist.sum() + 1e-12)

    return np.hstack([h_hist, s_hist, v_hist]).astype(np.float64)


def lbp_mean_std_2(gray: np.ndarray) -> np.ndarray:
    lbp = mahotas.features.lbp(gray, radius=1, points=8).astype(np.float64)
    return np.array([float(lbp.mean()), float(lbp.std())], dtype=np.float64)


def hu_7(mask: np.ndarray) -> np.ndarray:
    m = cv2.moments((mask > 0).astype(np.uint8))
    hu = cv2.HuMoments(m).flatten().astype(np.float64)
    out = np.zeros_like(hu)
    for i, v in enumerate(hu):
        if v == 0:
            out[i] = 0.0
        else:
            out[i] = -np.sign(v) * np.log10(abs(v))
    return out


def mean_std_hsv_6(segmented_rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2HSV)
    m = (mask > 0).astype(np.uint8)
    mean, std = cv2.meanStdDev(hsv, mask=m)
    mean = mean.flatten()[:3].astype(np.float64)
    std = std.flatten()[:3].astype(np.float64)
    return mean, std


def morphology_1(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    g = gray.copy()
    fg = mask > 0
    if fg.any():
        fg_mean = float(np.mean(g[fg]))
        g2 = g.astype(np.float64)
        g2[~fg] = fg_mean
        g = np.clip(g2, 0, 255).astype(np.uint8)

    _, thr = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k, iterations=1)

    if fg.any():
        val = float(np.mean(opened[fg]) / 255.0)
    else:
        val = float(opened.mean() / 255.0)
    return np.array([val], dtype=np.float64)
