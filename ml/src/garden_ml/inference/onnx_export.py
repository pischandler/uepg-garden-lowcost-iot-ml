"""Optional ONNX export and load. Requires pip install -e '.[onnx]'."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


def _register_xgboost_converter() -> None:
    """Register XGBoost converter for skl2onnx via onnxmltools (required for Pipeline export)."""
    try:
        from xgboost import XGBClassifier
        from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
        from skl2onnx import update_registered_converter
        from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes

        update_registered_converter(
            XGBClassifier,
            "XGBoostXGBClassifier",
            calculate_linear_classifier_output_shapes,
            convert_xgboost,
            options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
        )
        logger.debug("xgboost_converter_registered")
    except ImportError:
        logger.warning("onnxmltools_not_installed — XGBoost pipeline export may fail; install extras: pip install -e '.[onnx]'")


def export_pipeline_to_onnx(pipeline: Any, output_path: Path, n_features: int = 188) -> bool:
    """Export a sklearn Pipeline (scaler + model) to ONNX. Returns True on success."""
    try:
        import onnx
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        logger.debug("onnx/skl2onnx not installed; skip ONNX export")
        return False

    _register_xgboost_converter()

    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(
            pipeline,
            initial_types=initial_type,
            target_opset={"": 14, "ai.onnx.ml": 3},
            options={id(pipeline): {"zipmap": False}},
        )
        onnx.save_model(onnx_model, str(output_path))
        logger.info("onnx_exported path={}", output_path)
        return True
    except Exception:
        logger.exception("onnx_export_failed")
        return False


def verify_onnx_roundtrip(pipeline: Any, onnx_path: Path, n_features: int = 188, atol: float = 1e-5) -> bool:
    """Load the saved ONNX and compare predict_proba against sklearn on a random sample."""
    try:
        import onnxruntime as ort
    except ImportError:
        logger.debug("onnxruntime not installed; skip round-trip verification")
        return True

    rng = np.random.default_rng(0)
    X = rng.random((8, n_features)).astype(np.float32)

    proba_sklearn = pipeline.predict_proba(X.astype(np.float64))

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.get_outputs()
    out_name = outputs[1].name if len(outputs) > 1 else outputs[0].name
    proba_onnx = np.asarray(session.run([out_name], {input_name: X})[0], dtype=np.float64)

    if not np.allclose(proba_sklearn, proba_onnx, atol=atol):
        max_diff = float(np.max(np.abs(proba_sklearn - proba_onnx)))
        logger.error("onnx_roundtrip_failed max_diff={:.2e} atol={:.2e}", max_diff, atol)
        return False

    logger.info("onnx_roundtrip_ok max_diff={:.2e}", float(np.max(np.abs(proba_sklearn - proba_onnx))))
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Export sklearn pipeline to ONNX with round-trip verification.")
    ap.add_argument("--artifacts_dir", required=True, help="Path to model registry version dir (e.g. artifacts/model_registry/v0005)")
    ap.add_argument("--n_features", type=int, default=188)
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--no_verify", action="store_true", help="Skip round-trip verification")
    args = ap.parse_args()

    import joblib
    from garden_ml.config.constants import MODEL_FILE, TRAIN_META_FILE

    artifacts_dir = Path(args.artifacts_dir)
    model_path = artifacts_dir / MODEL_FILE
    onnx_path = artifacts_dir / "model.onnx"

    if not model_path.exists():
        logger.error("model_not_found path={}", model_path)
        return 1

    pipeline = joblib.load(model_path)
    ok = export_pipeline_to_onnx(pipeline, onnx_path, n_features=args.n_features)
    if not ok:
        return 1

    if not args.no_verify:
        ok = verify_onnx_roundtrip(pipeline, onnx_path, n_features=args.n_features, atol=args.atol)
        if not ok:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
