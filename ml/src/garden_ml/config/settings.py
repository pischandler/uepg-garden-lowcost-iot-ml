from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GML_", env_file=".env", extra="ignore")

    img_size: int = Field(default=128)
    img_size_dl: int = Field(default=224)

    save_debug: bool = Field(default=False)
    debug_root: Path = Field(default=Path("salvas"))

    max_content_length: int = Field(default=8 * 1024 * 1024)
    topk: int = Field(default=3)

    min_input_side_px: int = Field(default=64)

    # thresholds de confiança/qualidade (server-side)
    min_confidence: float = Field(default=0.60)
    min_mask_coverage: float = Field(default=0.08)

    # novos gates de qualidade (baseados na imagem segmentada)
    # 0 = desativado
    min_mean_v: float = Field(default=0.0)           # brilho médio no canal V (HSV) na máscara
    min_laplacian_var: float = Field(default=0.0)    # nitidez (variância do laplaciano)

    artifacts_dir: Path = Field(default=Path("artifacts") / "model_registry" / "v0004")
    reports_dir: Path = Field(default=Path("reports"))

    mlflow_enabled: bool = Field(default=True)
    mlflow_tracking_uri: str = Field(default="file:./mlruns")
    mlflow_experiment: str = Field(default="smart-tomato-garden")

    # treino: controle do paralelismo do GridSearch (evita OOM em container)
    grid_n_jobs: int = Field(default=1)
    grid_pre_dispatch: str = Field(default="1*n_jobs")

    # opcional: dump grande (não é necessário pro pipeline funcionar)
    dump_features_csv: bool = Field(default=False)

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=5000)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    seed: int = Field(default=42)

    def ensure_dirs(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        if self.save_debug:
            self.debug_root.mkdir(parents=True, exist_ok=True)
