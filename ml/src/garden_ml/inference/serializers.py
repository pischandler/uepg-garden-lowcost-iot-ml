from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PredictionTopK:
    classe: str
    score: float


@dataclass(frozen=True)
class PredictionResponse:
    classe_predita: str
    score: float
    topk: list[PredictionTopK]
    timings_ms: dict[str, float]
    meta: dict[str, Any]
