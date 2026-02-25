from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class PredictionTopK(BaseModel):
    classe: str
    score: float


class PredictionResponse(BaseModel):
    classe_predita: str | None
    score: float
    topk: list[PredictionTopK]
    timings_ms: dict[str, float]
    quality: dict[str, Any]
    meta: dict[str, Any]

    # qualidade/confiança consolidada
    confident: bool
    reasons: list[str]

    # thresholds usados
    min_confidence: float
    min_mask_coverage: float
    min_mean_v: float
    min_laplacian_var: float

    photometric_normalize_used: bool
    model_img_size: int
