from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import confusion_matrix

from garden_ml.config.constants import ENCODER_FILE, MODEL_FILE, TRAIN_META_FILE
from garden_ml.config.logging import setup_logging
from garden_ml.data.manifest import samples_from_manifest, scan_folder_dataset, write_json
from garden_ml.features.extract import ExtractOptions, extract_features_from_path, extract_features_from_rgb
from garden_ml.image.photometric import brightness_contrast_rgb, gamma_rgb
from garden_ml.training.train import compute_metrics


def load_artifacts(artifacts_dir: Path) -> tuple[Any, Any]:
    model = joblib.load(artifacts_dir / MODEL_FILE)
    le = joblib.load(artifacts_dir / ENCODER_FILE)
    return model, le


def load_training_meta(artifacts_dir: Path) -> dict[str, Any]:
    p = artifacts_dir / TRAIN_META_FILE
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    n = float(len(conf))
    if n <= 0:
        return 0.0

    for i in range(int(n_bins)):
        lo = bins[i]
        hi = bins[i + 1]
        if i == int(n_bins) - 1:
            m = (conf >= lo) & (conf <= hi)
        else:
            m = (conf >= lo) & (conf < hi)
        if not bool(np.any(m)):
            continue
        frac = float(np.sum(m)) / n
        acc = float(np.mean(correct[m]))
        avg_conf = float(np.mean(conf[m]))
        ece += frac * abs(acc - avg_conf)

    return float(ece)


def _load_test_manifest_csv(p: Path) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    with p.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "path" not in r.fieldnames or "class" not in r.fieldnames:
            raise ValueError("invalid test_manifest.csv")
        for row in r:
            path = (row.get("path") or "").strip()
            cls = (row.get("class") or "").strip()
            if path and cls:
                out.append((Path(path), cls))
    if not out:
        raise ValueError("empty test_manifest.csv")
    return out


def _resolve_eval_items(dataset_dir: Path, artifacts_dir: Path, manifest: str) -> list[tuple[Path, str]]:
    test_manifest = artifacts_dir / "test_manifest.csv"
    if test_manifest.is_file():
        return _load_test_manifest_csv(test_manifest)

    try:
        rows = samples_from_manifest(dataset_dir, manifest, include_kinds={"orig"}, require_status_ok=True)
        return [(Path(p), cls) for p, cls, _gid, _kind in rows]
    except Exception:
        raw = scan_folder_dataset(dataset_dir)
        return [(p, c) for p, c, _g in raw]


def eval_dataset(dataset_dir: Path, artifacts_dir: Path, img_size: int, manifest: str, normalize: bool) -> dict[str, Any]:
    items = _resolve_eval_items(dataset_dir, artifacts_dir, manifest)

    model, le = load_artifacts(artifacts_dir)
    opts = ExtractOptions(img_size=img_size, photometric_normalize=normalize)

    X_list: list[np.ndarray] = []
    y_list: list[str] = []
    times_feat: list[float] = []
    for p, c in items:
        t0 = time.perf_counter()
        feat = extract_features_from_path(p, opts)
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
    pred_ms = (t3 - t2) * 1000.0 / max(1, int(X.shape[0]))

    metrics = compute_metrics(y_enc, y_pred, labels=list(le.classes_))
    cm = confusion_matrix(y_enc, y_pred)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        metrics["ece_15"] = expected_calibration_error(np.asarray(probs, dtype=np.float64), y_enc, n_bins=15)
        metrics["avg_confidence"] = float(np.mean(np.max(probs, axis=1)))
    else:
        metrics["ece_15"] = 0.0
        metrics["avg_confidence"] = 0.0

    latency = {
        "feature_ms_mean": float(np.mean(times_feat)),
        "feature_ms_p50": float(np.percentile(times_feat, 50)),
        "feature_ms_p95": float(np.percentile(times_feat, 95)),
        "predict_ms_per_sample": float(pred_ms),
        "total_ms_per_sample_est": float(np.mean(times_feat) + pred_ms),
    }

    return {
        "normalize": bool(normalize),
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "classes": list(le.classes_),
        "latency": latency,
        "img_size_used": int(img_size),
    }


def illumination_sensitivity(dataset_dir: Path, artifacts_dir: Path, img_size: int, manifest: str) -> dict[str, Any]:
    from PIL import Image

    items = _resolve_eval_items(dataset_dir, artifacts_dir, manifest)
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
            x = gamma_rgb(rgb, gamma) if float(gamma) != 1.0 else rgb
            x = brightness_contrast_rgb(x, alpha=float(alpha), beta=float(beta)) if (float(alpha) != 1.0 or float(beta) != 0.0) else x
            feat = extract_features_from_rgb(x, opts)
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
    setup_logging()

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, default="dataset_aug")
    ap.add_argument("--artifacts_dir", type=str, default="artifacts/model_registry/v0004")
    ap.add_argument("--img_size", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manifest", type=str, default="augmentation_manifest.csv")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    artifacts_dir = Path(args.artifacts_dir)
    out_dir = artifacts_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_training_meta(artifacts_dir)
    trained_norm = bool(meta.get("photometric_normalize", False))
    meta_size = int(meta.get("img_size", 128))
    img_size = int(args.img_size) if int(args.img_size) > 0 else int(meta_size)

    logger.info(
        "eval_start dataset_dir={} artifacts_dir={} img_size={} manifest={} trained_photometric_normalize={}",
        str(dataset_dir),
        str(artifacts_dir),
        int(img_size),
        str(args.manifest),
        bool(trained_norm),
    )

    eval_trained = eval_dataset(dataset_dir, artifacts_dir, img_size=int(img_size), manifest=args.manifest, normalize=trained_norm)
    eval_base = eval_dataset(dataset_dir, artifacts_dir, img_size=int(img_size), manifest=args.manifest, normalize=False)
    eval_norm = eval_dataset(dataset_dir, artifacts_dir, img_size=int(img_size), manifest=args.manifest, normalize=True)
    sens = illumination_sensitivity(dataset_dir, artifacts_dir, img_size=int(img_size), manifest=args.manifest)

    write_json(out_dir / "eval_trained.json", eval_trained)
    write_json(out_dir / "eval_base.json", eval_base)
    write_json(out_dir / "eval_normalized.json", eval_norm)
    write_json(out_dir / "illumination_sensitivity.json", sens)

    pd.DataFrame(eval_base["metrics"]["per_class"]).to_csv(out_dir / "per_class_base.csv", index=False)
    pd.DataFrame(eval_norm["metrics"]["per_class"]).to_csv(out_dir / "per_class_normalized.csv", index=False)

    logger.info("eval_done output_dir={}", str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
