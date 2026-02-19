from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class DebugStorage:
    root: Path

    def ensure(self) -> None:
        (self.root / "original").mkdir(parents=True, exist_ok=True)
        (self.root / "csv").mkdir(parents=True, exist_ok=True)

    def csv_path(self) -> Path:
        return self.root / "csv" / "inferences.csv"

    def ensure_csv(self) -> None:
        p = self.csv_path()
        if p.exists():
            return
        with p.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp", "device_id", "source", "classe", "score", "total_ms", "lux_raw", "soil_raw", "temp_c", "hum_pct"])

    def save_original_bgr(self, bgr: np.ndarray, base: str) -> None:
        cv2.imwrite(str(self.root / "original" / f"{base}.jpg"), bgr)

    def append_row(
        self,
        device_id: str,
        source: str,
        classe: str,
        score: float,
        total_ms: float,
        lux_raw: str,
        soil_raw: str,
        temp_c: str,
        hum_pct: str,
    ) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.csv_path().open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ts, device_id, source, classe, score, total_ms, lux_raw, soil_raw, temp_c, hum_pct])
