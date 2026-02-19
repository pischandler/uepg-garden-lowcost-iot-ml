from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import confusion_matrix

from garden_ml.config.constants import ENCODER_FILE, MODEL_FILE
from garden_ml.data.manifest import samples_from_manifest, scan_folder_dataset, write_json
from garden_ml.features.extract import ExtractOptions, extract_102_from_path, extract_102_from_rgb
from garden_ml.image.photometric import brightness_contrast_rgb, gamma_rgb
from garden_ml.training.train import compute_metrics


def load_artifacts(artifacts_dir: Path) -> tuple[Any, Any]:
    model = joblib.load(artifacts_dir / MODEL_FILE)
    le = joblib.load(artifacts_dir / ENCODER_FILE)
    return model, le


def eval_dataset(dataset_dir: Path, artifacts_dir: Path, img_size: int, manifest: str, normalize: bool) -> dict[str, Any]:
    try:
        rows = samples_from_manifest(dataset_dir, manifest, include_kinds={"orig"}, require_status_ok=True)
        items = [(Path(p), cls) for p, cls, _gid, _kind in rows]
    except Exception:
        raw = scan_folder_dataset(dataset_dir)
        items = [(p, c) for p, c, _g in raw]

    model, le = load_artifacts(artifacts_dir)
    opts = ExtractOptions(img_size=img_size, photometric_normalize=normalize)

    X_list: list[np.ndarray] = []
    y_list: list[str] = []
    times_feat: list[float] = []
    for p, c in items:
        t0 = time.perf_counter()
        feat = extract_102_from_path(p, opts)
        t1 = time.perf_counter()
        X_list.append(feat)
        y_list.append(c)
        times_feat.append((t1 - t0) * 1000.0)

    X = np.vstack(X_list).astype(np.float64)
    y = np.array(y_list, dtype=object)
    y_enc = le.transform(y)

    t2 = time.perf_counter()
    y_pred = model.predict(X)
    t3 = time.perf_counter()
    pred_ms = (t3 - t2) * 1000.0 / max(1, X.shape[0])

    metrics = compute_metrics(y_enc, y_pred, labels=list(le.classes_))
    cm = confusion_matrix(y_enc, y_pred)

    latency = {
        "feature_ms_mean": float(np.mean(times_feat)),
        "feature_ms_p50": float(np.percentile(times_feat, 50)),
        "feature_ms_p95": float(np.percentile(times_feat, 95)),
        "predict_ms_per_sample": float(pred_ms),
        "total_ms_per_sample_est": float(np.mean(times_feat) + pred_ms),
    }

    return {"metrics": metrics, "confusion_matrix": cm.tolist(), "classes": list(le.classes_), "latency": latency}


def illumination_sensitivity(dataset_dir: Path, artifacts_dir: Path, img_size: int, manifest: str) -> dict[str, Any]:
    from PIL import Image

    try:
        rows = samples_from_manifest(dataset_dir, manifest, include_kinds={"orig"}, require_status_ok=True)
        items = [(Path(p), cls) for p, cls, _gid, _kind in rows]
    except Exception:
        raw = scan_folder_dataset(dataset_dir)
        items = [(p, c) for p, c, _g in raw]

    model, le = load_artifacts(artifacts_dir)

    scenarios = [
        {"name": "base", "gamma": 1.0, "alpha": 1.0, "beta": 0.0},
        {"name": "dark_gamma_1.4", "gamma": 1.4, "alpha": 1.0, "beta": 0.0},
        {"name": "bright_gamma_0.8", "gamma": 0.8, "alpha": 1.0, "beta": 0.0},
        {"name": "contrast_low", "gamma": 1.0, "alpha": 0.85, "beta": 0.0},
        {"name": "contrast_high", "gamma": 1.0, "alpha": 1.15, "beta": 0.0},
        {"name": "brightness_minus20", "gamma": 1.0, "alpha": 1.0, "beta": -20.0},
        {"name": "brightness_plus20", "gamma": 1.0, "alpha": 1.0, "beta": 20.0},
    ]

    def run_scenario(normalize: bool, name: str, gamma: float, alpha: float, beta: float) -> dict[str, Any]:
        opts = ExtractOptions(img_size=img_size, photometric_normalize=normalize)

        X_list: list[np.ndarray] = []
        y_list: list[str] = []
        for p, c in items:
            rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
            x = gamma_rgb(rgb, gamma) if gamma != 1.0 else rgb
            x = brightness_contrast_rgb(x, alpha=alpha, beta=beta) if (alpha != 1.0 or beta != 0.0) else x
            feat = extract_102_from_rgb(x, opts)
            X_list.append(feat)
            y_list.append(c)

        X = np.vstack(X_list).astype(np.float64)
        y = np.array(y_list, dtype=object)
        y_enc = le.transform(y)
        y_pred = model.predict(X)
        m = compute_metrics(y_enc, y_pred, labels=list(le.classes_))
        return {"scenario": name, "metrics": {k: v for k, v in m.items() if k != "per_class"}, "per_class": m["per_class"]}

    base = []
    norm = []
    for s in scenarios:
        base.append(run_scenario(False, s["name"], s["gamma"], s["alpha"], s["beta"]))
        norm.append(run_scenario(True, s["name"], s["gamma"], s["alpha"], s["beta"]))

    return {"base": base, "normalized": norm, "scenarios": scenarios}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, default="dataset_aug")
    ap.add_argument("--artifacts_dir", type=str, default="artifacts/model_registry/v0001")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manifest", type=str, default="augmentation_manifest.csv")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    artifacts_dir = Path(args.artifacts_dir)
    out_dir = artifacts_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_base = eval_dataset(dataset_dir, artifacts_dir, img_size=int(args.img_size), manifest=args.manifest, normalize=False)
    eval_norm = eval_dataset(dataset_dir, artifacts_dir, img_size=int(args.img_size), manifest=args.manifest, normalize=True)
    sens = illumination_sensitivity(dataset_dir, artifacts_dir, img_size=int(args.img_size), manifest=args.manifest)

    write_json(out_dir / "eval_base.json", eval_base)
    write_json(out_dir / "eval_normalized.json", eval_norm)
    write_json(out_dir / "illumination_sensitivity.json", sens)

    pd.DataFrame(eval_base["metrics"]["per_class"]).to_csv(out_dir / "per_class_base.csv", index=False)
    pd.DataFrame(eval_norm["metrics"]["per_class"]).to_csv(out_dir / "per_class_normalized.csv", index=False)

    logger.info("saved evaluation at {}", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
