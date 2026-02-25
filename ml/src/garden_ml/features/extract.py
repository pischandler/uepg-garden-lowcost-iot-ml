from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from garden_ml.features.components import (
    chroma_rg_mean_std_4,
    haralick_13,
    hsv_hist_48,
    hu_7,
    lab_ab_hist_16,
    lab_ab_mean_std_4,
    lbp_nri_uniform_hist_59,
    mean_std_hsv_6,
    shape_basic_6,
    zernike_25,
)
from garden_ml.features.schema import validate_dim
from garden_ml.image.photometric import normalize_pipeline_rgb
from garden_ml.image.segmentation import segment_leaf_hsv


@dataclass(frozen=True)
class ExtractOptions:
    img_size: int
    photometric_normalize: bool = False


def _fill_background_gray(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    g = gray.astype(np.uint8, copy=True)
    fg = mask > 0
    if bool(np.any(fg)):
        v = float(np.mean(g[fg]))
        g[~fg] = int(round(v))
    return g


def _mean_v_hsv(segmented_rgb: np.ndarray, mask: np.ndarray) -> float:
    hsv = cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]
    fg = mask > 0
    vals = v[fg] if bool(np.any(fg)) else v.reshape(-1)
    return float(np.mean(vals)) if int(vals.size) > 0 else 0.0


def _laplacian_var(gray: np.ndarray, mask: np.ndarray) -> float:
    g = gray.astype(np.uint8, copy=False)
    lap = cv2.Laplacian(g, cv2.CV_64F)
    fg = mask > 0
    vals = lap[fg] if bool(np.any(fg)) else lap.reshape(-1)
    return float(np.var(vals)) if int(vals.size) > 0 else 0.0


def extract_features_and_meta_from_rgb(rgb: np.ndarray, opts: ExtractOptions) -> tuple[np.ndarray, dict[str, Any]]:
    segmented, mask = segment_leaf_hsv(rgb, size=(opts.img_size, opts.img_size))

    if opts.photometric_normalize:
        segmented = normalize_pipeline_rgb(segmented, mask=mask)

    gray = cv2.cvtColor(segmented, cv2.COLOR_RGB2GRAY)
    gray_tex = _fill_background_gray(gray, mask)

    har = haralick_13(gray_tex)
    zer = zernike_25(mask)
    hsvh = hsv_hist_48(segmented, mask)
    lbph = lbp_nri_uniform_hist_59(gray_tex, mask)
    hu = hu_7(mask)
    mean_hsv, std_hsv = mean_std_hsv_6(segmented, mask)
    shp = shape_basic_6(mask)
    labms = lab_ab_mean_std_4(segmented, mask)
    labh = lab_ab_hist_16(segmented, mask)
    chroma = chroma_rg_mean_std_4(segmented, mask)

    feat = np.hstack([har, zer, hsvh, lbph, hu, mean_hsv, std_hsv, shp, labms, labh, chroma]).astype(np.float64)
    validate_dim(int(feat.size))

    nz = int(np.count_nonzero(mask))
    total = int(mask.size)
    coverage = float(nz / max(1, total))

    mean_v = _mean_v_hsv(segmented, mask)
    lap_var = _laplacian_var(gray_tex, mask)

    meta = {
        "mask_nonzero": nz,
        "mask_total": total,
        "mask_coverage": coverage,
        "mean_v": float(mean_v),
        "laplacian_var": float(lap_var),
    }
    return feat, meta


def extract_features_from_rgb(rgb: np.ndarray, opts: ExtractOptions) -> np.ndarray:
    feat, _meta = extract_features_and_meta_from_rgb(rgb, opts)
    return feat


def extract_features_from_path(path: Path, opts: ExtractOptions) -> np.ndarray:
    import PIL.Image

    pil = PIL.Image.open(path).convert("RGB")
    rgb = np.asarray(pil, dtype=np.uint8)
    return extract_features_from_rgb(rgb, opts)
