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

    artifacts_dir: Path = Field(default=Path("artifacts") / "model_registry" / "v0001")
    reports_dir: Path = Field(default=Path("reports"))

    mlflow_enabled: bool = Field(default=True)
    mlflow_tracking_uri: str = Field(default="file:./mlruns")
    mlflow_experiment: str = Field(default="smart-tomato-garden")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=5000)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    seed: int = Field(default=42)

    def ensure_dirs(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        if self.save_debug:
            self.debug_root.mkdir(parents=True, exist_ok=True)
