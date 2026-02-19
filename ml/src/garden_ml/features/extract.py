from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from garden_ml.features.components import (
    haralick_13,
    hsv_hist_48,
    hu_7,
    lbp_mean_std_2,
    mean_std_hsv_6,
    morphology_1,
    zernike_25,
)
from garden_ml.features.schema import validate_dim
from garden_ml.image.photometric import normalize_pipeline_rgb
from garden_ml.image.segmentation import segment_leaf_hsv


@dataclass(frozen=True)
class ExtractOptions:
    img_size: int
    photometric_normalize: bool = False


def extract_102_from_rgb(rgb: np.ndarray, opts: ExtractOptions) -> np.ndarray:
    if opts.photometric_normalize:
        rgb = normalize_pipeline_rgb(rgb)

    segmented, mask = segment_leaf_hsv(rgb, size=(opts.img_size, opts.img_size))
    gray = cv2.cvtColor(segmented, cv2.COLOR_RGB2GRAY)

    har = haralick_13(gray)
    zer = zernike_25(mask)
    hsvh = hsv_hist_48(segmented, mask)
    lbp = lbp_mean_std_2(gray)
    hu = hu_7(mask)
    mean_hsv, std_hsv = mean_std_hsv_6(segmented, mask)
    morph = morphology_1(gray, mask)

    feat = np.hstack([har, zer, hsvh, lbp, hu, mean_hsv, std_hsv, morph]).astype(np.float64)
    validate_dim(int(feat.size))
    return feat


def extract_102_from_path(path: Path, opts: ExtractOptions) -> np.ndarray:
    import PIL.Image

    pil = PIL.Image.open(path).convert("RGB")
    rgb = np.asarray(pil, dtype=np.uint8)
    return extract_102_from_rgb(rgb, opts)
