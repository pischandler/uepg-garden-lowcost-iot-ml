from __future__ import annotations

import os
import sys

from loguru import logger


def setup_logging(level: str | None = None) -> None:
    lvl = (level or os.getenv("GML_LOG_LEVEL", "INFO")).upper().strip()
    logger.remove()
    logger.add(sys.stderr, level=lvl, enqueue=False, backtrace=False, diagnose=False)
