from __future__ import annotations

import math

import cv2
import mahotas
import numpy as np
from skimage.feature import local_binary_pattern


def haralick_13(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.uint8, copy=False)
    return mahotas.features.haralick(g).mean(axis=0).astype(np.float64)


def zernike_25(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    radius = max(1, min(m.shape[0], m.shape[1]) // 2)
    z = mahotas.features.zernike_moments(m, radius, degree=8).astype(np.float64)
    if int(z.size) != 25:
        raise ValueError(f"zernike size {int(z.size)} != 25")
    return z


def hsv_hist_48(segmented_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2HSV)
    m = (mask > 0).astype(np.uint8)
    use_mask = None if int(np.count_nonzero(m)) == 0 else m

    h_hist = cv2.calcHist([hsv], [0], use_mask, [16], [0, 180]).flatten().astype(np.float64)
    s_hist = cv2.calcHist([hsv], [1], use_mask, [16], [0, 256]).flatten().astype(np.float64)
    v_hist = cv2.calcHist([hsv], [2], use_mask, [16], [0, 256]).flatten().astype(np.float64)

    h_hist /= (float(h_hist.sum()) + 1e-12)
    s_hist /= (float(s_hist.sum()) + 1e-12)
    v_hist /= (float(v_hist.sum()) + 1e-12)

    return np.hstack([h_hist, s_hist, v_hist]).astype(np.float64)


def lbp_nri_uniform_hist_59(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    g = gray.astype(np.uint8, copy=False)
    lbp = local_binary_pattern(g, P=8, R=1, method="nri_uniform")
    fg = mask > 0
    vals = lbp[fg] if bool(np.any(fg)) else lbp.reshape(-1)
    n_bins = 59
    hist, _ = np.histogram(vals, bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float64)
    hist /= (float(hist.sum()) + 1e-12)
    return hist


def hu_7(mask: np.ndarray) -> np.ndarray:
    m = cv2.moments((mask > 0).astype(np.uint8))
    hu = cv2.HuMoments(m).flatten().astype(np.float64)
    out = np.zeros_like(hu)
    for i, v in enumerate(hu):
        if float(v) == 0.0:
            out[i] = 0.0
        else:
            out[i] = -np.sign(v) * np.log10(abs(v))
    return out


def mean_std_hsv_6(segmented_rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2HSV)
    m = (mask > 0).astype(np.uint8)
    use_mask = None if int(np.count_nonzero(m)) == 0 else m
    mean, std = cv2.meanStdDev(hsv, mask=use_mask)
    mean = mean.flatten()[:3].astype(np.float64)
    std = std.flatten()[:3].astype(np.float64)
    return mean, std


def shape_basic_6(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8) * 255
    h, w = m.shape[:2]
    area_total = float(h * w)
    fg_area = float(np.count_nonzero(m))

    if fg_area <= 0.0:
        return np.zeros((6,), dtype=np.float64)

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros((6,), dtype=np.float64)

    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    per = float(cv2.arcLength(c, True))
    x, y, bw, bh = cv2.boundingRect(c)
    bbox_area = float(bw * bh)

    hull = cv2.convexHull(c)
    hull_area = float(cv2.contourArea(hull))

    area_ratio = area / (area_total + 1e-12)
    per_ratio = per / (2.0 * (h + w) + 1e-12)
    aspect = float(bw) / (float(bh) + 1e-12)
    extent = area / (bbox_area + 1e-12)
    solidity = area / (hull_area + 1e-12)
    circularity = (4.0 * math.pi * area) / (per * per + 1e-12)

    return np.array([area_ratio, per_ratio, aspect, extent, solidity, circularity], dtype=np.float64)


def lab_ab_mean_std_4(segmented_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2LAB)
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    fg = mask > 0
    if not bool(np.any(fg)):
        a_vals = a.reshape(-1).astype(np.float64)
        b_vals = b.reshape(-1).astype(np.float64)
    else:
        a_vals = a[fg].astype(np.float64)
        b_vals = b[fg].astype(np.float64)

    am = float(np.mean(a_vals))
    ast = float(np.std(a_vals))
    bm = float(np.mean(b_vals))
    bst = float(np.std(b_vals))
    return np.array([am, ast, bm, bst], dtype=np.float64)


def lab_ab_hist_16(segmented_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2LAB)
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    fg = mask > 0
    if not bool(np.any(fg)):
        a_vals = a.reshape(-1)
        b_vals = b.reshape(-1)
    else:
        a_vals = a[fg]
        b_vals = b[fg]

    bins = 8
    ha, _ = np.histogram(a_vals, bins=bins, range=(0, 256))
    hb, _ = np.histogram(b_vals, bins=bins, range=(0, 256))
    ha = ha.astype(np.float64)
    hb = hb.astype(np.float64)
    ha /= (float(ha.sum()) + 1e-12)
    hb /= (float(hb.sum()) + 1e-12)
    return np.hstack([ha, hb]).astype(np.float64)


def chroma_rg_mean_std_4(segmented_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = segmented_rgb.astype(np.float64)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    s = r + g + b + 1e-12
    rc = r / s
    gc = g / s

    fg = mask > 0
    if bool(np.any(fg)):
        rc_v = rc[fg]
        gc_v = gc[fg]
    else:
        rc_v = rc.reshape(-1)
        gc_v = gc.reshape(-1)

    rcm = float(np.mean(rc_v))
    rcs = float(np.std(rc_v))
    gcm = float(np.mean(gc_v))
    gcs = float(np.std(gc_v))
    return np.array([rcm, rcs, gcm, gcs], dtype=np.float64)
