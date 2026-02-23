from __future__ import annotations

import cv2
import numpy as np

from garden_ml.image.resize import letterbox_rgb


def segment_leaf_hsv(rgb: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    rgb = letterbox_rgb(rgb, size=size, bg=0)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    lower1 = np.array([15, 25, 25], dtype=np.uint8)
    upper1 = np.array([40, 255, 255], dtype=np.uint8)
    lower2 = np.array([40, 25, 25], dtype=np.uint8)
    upper2 = np.array([95, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    area_total = int(mask.shape[0] * mask.shape[1])
    min_area = max(64, int(area_total * 0.01))

    num, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, int(num)):
        if int(stats[i, cv2.CC_STAT_AREA]) >= int(min_area):
            keep[lab == i] = 255

    if int(np.count_nonzero(keep)) == 0:
        keep[:] = 255
        return rgb, keep

    keep = cv2.GaussianBlur(keep, (5, 5), 0)
    keep = (keep > 0).astype(np.uint8) * 255
    segmented = cv2.bitwise_and(rgb, rgb, mask=keep)
    return segmented, keep
