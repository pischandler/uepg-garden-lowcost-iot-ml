from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from garden_ml.config.constants import FEATURES_DIM


@dataclass(frozen=True)
class FeatureSchema:
    total: int
    parts: dict[str, int]
    order: list[str]

    def as_dict(self) -> dict:
        return {"total": self.total, "parts": self.parts, "order": self.order}

    def sha1(self) -> str:
        raw = json.dumps(self.as_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()


SCHEMA = FeatureSchema(
    total=FEATURES_DIM,
    parts={
        "haralick_13": 13,
        "zernike_25_degree8": 25,
        "hsv_hist_48": 48,
        "lbp_mean_std_2": 2,
        "hu_7": 7,
        "mean_hsv_3": 3,
        "std_hsv_3": 3,
        "morphology_1": 1,
    },
    order=[
        "haralick_13",
        "zernike_25_degree8",
        "hsv_hist_48",
        "lbp_mean_std_2",
        "hu_7",
        "mean_hsv_3",
        "std_hsv_3",
        "morphology_1",
    ],
)


def validate_dim(n: int) -> None:
    if n != FEATURES_DIM:
        raise ValueError(f"feature schema expects {FEATURES_DIM}, got {n}")


def write_schema(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(SCHEMA.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
