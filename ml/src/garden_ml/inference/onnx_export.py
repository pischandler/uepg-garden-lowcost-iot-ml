"""Optional ONNX export and load. Requires pip install -e '.[onnx]'."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger


def export_pipeline_to_onnx(pipeline: Any, output_path: Path, n_features: int = 188) -> bool:
    """Export a sklearn Pipeline (scaler + model) to ONNX. Returns True on success."""
    try:
        import onnx
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        logger.debug("onnx/skl2onnx not installed; skip ONNX export")
        return False

    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(
            pipeline,
            initial_types=initial_type,
            target_opset=14,
            options={id(pipeline): {"zipmap": False}},
        )
        onnx.save_model(onnx_model, str(output_path))
        logger.info("onnx_exported path={}", output_path)
        return True
    except Exception:
        logger.exception("onnx_export_failed")
        return False
