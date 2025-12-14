import argparse
import csv
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import mahotas
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
from xgboost import XGBClassifier


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def letterbox_rgb(rgb: np.ndarray, size: tuple[int, int], bg: int = 0) -> np.ndarray:
    h, w = rgb.shape[:2]
    th, tw = size
    scale = min(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((th, tw, 3), bg, dtype=np.uint8)
    top = (th - nh) // 2
    left = (tw - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas


def segment_leaf_hsv(rgb: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    rgb = letterbox_rgb(rgb, size=size, bg=0)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    lower1 = np.array([15, 25, 25], dtype=np.uint8)
    upper1 = np.array([40, 255, 255], dtype=np.uint8)
    lower2 = np.array([40, 25, 25], dtype=np.uint8)
    upper2 = np.array([95, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    area_total = mask.shape[0] * mask.shape[1]
    min_area = max(64, int(area_total * 0.01))

    num, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 255

    keep = cv2.GaussianBlur(keep, (5, 5), 0)
    keep = (keep > 0).astype(np.uint8) * 255
    segmented = cv2.bitwise_and(rgb, rgb, mask=keep)
    return segmented, keep


def hsv_hist_48(segmented_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2HSV)
    m = (mask > 0).astype(np.uint8)

    h_hist = cv2.calcHist([hsv], [0], m, [16], [0, 180]).flatten().astype(np.float64)
    s_hist = cv2.calcHist([hsv], [1], m, [16], [0, 256]).flatten().astype(np.float64)
    v_hist = cv2.calcHist([hsv], [2], m, [16], [0, 256]).flatten().astype(np.float64)

    h_hist /= (h_hist.sum() + 1e-12)
    s_hist /= (s_hist.sum() + 1e-12)
    v_hist /= (v_hist.sum() + 1e-12)

    return np.hstack([h_hist, s_hist, v_hist]).astype(np.float64)


def hu_moments_7(mask: np.ndarray) -> np.ndarray:
    m = cv2.moments((mask > 0).astype(np.uint8))
    hu = cv2.HuMoments(m).flatten().astype(np.float64)
    out = np.zeros_like(hu)
    for i, v in enumerate(hu):
        if v == 0:
            out[i] = 0.0
        else:
            out[i] = -np.sign(v) * np.log10(abs(v))
    return out


def morphology_feature_1(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    g = gray.copy()
    fg = mask > 0
    if fg.any():
        fg_mean = float(np.mean(g[fg]))
        g = g.astype(np.float64)
        g[~fg] = fg_mean
        g = np.clip(g, 0, 255).astype(np.uint8)

    _, thresh = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k, iterations=1)

    if fg.any():
        val = float(np.mean(opened[fg]) / 255.0)
    else:
        val = float(opened.mean() / 255.0)

    return np.array([val], dtype=np.float64)


def extract_features_102(img_path: Path, img_size: tuple[int, int]) -> np.ndarray:
    pil = Image.open(img_path).convert("RGB")
    rgb = np.array(pil, dtype=np.uint8)

    segmented, mask = segment_leaf_hsv(rgb, size=img_size)
    gray = cv2.cvtColor(segmented, cv2.COLOR_RGB2GRAY)

    haralick = mahotas.features.haralick(gray).mean(axis=0).astype(np.float64)

    radius = min(mask.shape[0], mask.shape[1]) // 2
    zernike = mahotas.features.zernike_moments((mask > 0).astype(np.uint8), radius, degree=8).astype(np.float64)
    if zernike.size != 25:
        raise ValueError(f"zernike size {zernike.size} != 25")

    lbp_vec = mahotas.features.lbp(gray, radius=1, points=8).astype(np.float64)
    lbp_mean = float(lbp_vec.mean())
    lbp_std = float(lbp_vec.std())
    lbp_2 = np.array([lbp_mean, lbp_std], dtype=np.float64)

    hu = hu_moments_7(mask)

    hsv = cv2.cvtColor(segmented, cv2.COLOR_RGB2HSV)
    mean_hsv, std_hsv = cv2.meanStdDev(hsv, mask=(mask > 0).astype(np.uint8))
    mean_hsv = mean_hsv.flatten()[:3].astype(np.float64)
    std_hsv = std_hsv.flatten()[:3].astype(np.float64)

    hsv_hist = hsv_hist_48(segmented, mask)
    morph = morphology_feature_1(gray, mask)

    feat = np.hstack([haralick, zernike, hsv_hist, lbp_2, hu, mean_hsv, std_hsv, morph]).astype(np.float64)
    if feat.size != 102:
        raise ValueError(f"feature vector has {feat.size} values, expected 102")
    return feat


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    param_grid: dict


def build_model_specs(seed: int) -> list[ModelSpec]:
    rf = RandomForestClassifier(random_state=seed, n_jobs=-1)

    svm = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(probability=True, random_state=seed)),
        ]
    )

    xgb = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )

    return [
        ModelSpec(
            name="RandomForest",
            estimator=rf,
            param_grid={
                "n_estimators": [100, 200, 400],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt"],
            },
        ),
        ModelSpec(
            name="SVM",
            estimator=svm,
            param_grid={
                "svc__kernel": ["rbf"],
                "svc__C": [0.1, 1, 10],
                "svc__gamma": ["scale", "auto"],
            },
        ),
        ModelSpec(
            name="XGBoost",
            estimator=xgb,
            param_grid={
                "n_estimators": [100, 200],
                "max_depth": [3, 6, 10],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        ),
    ]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_samples_from_manifest(dataset_dir: Path, manifest_path: Path) -> list[tuple[Path, str, str]]:
    rows: list[tuple[Path, str, str]] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"class", "group_id", "output_path", "status", "kind"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"manifest columns invalid: {reader.fieldnames}")
        for r in reader:
            if (r.get("status") or "").strip().lower() != "ok":
                continue
            kind = (r.get("kind") or "").strip().lower()
            if kind not in {"orig", "aug"}:
                continue
            c = (r.get("class") or "").strip()
            g = (r.get("group_id") or "").strip()
            op = (r.get("output_path") or "").strip()
            if not c or not g or not op:
                continue
            p = dataset_dir / op
            if p.is_file() and is_image_file(p):
                rows.append((p, c, g))
    return rows


def scan_samples(dataset_dir: Path) -> list[tuple[Path, str, str]]:
    rows: list[tuple[Path, str, str]] = []
    classes = sorted([p.name for p in dataset_dir.iterdir() if p.is_dir()])
    for c in classes:
        class_dir = dataset_dir / c
        for p in sorted([x for x in class_dir.iterdir() if x.is_file() and is_image_file(x)]):
            stem = p.stem
            base = stem.split("__")[0] if "__" in stem else stem
            g = f"{c}/{base}"
            rows.append((p, c, g))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="dataset_aug")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--test_size", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--scoring", type=str, default="accuracy")
    parser.add_argument("--manifest", type=str, default="augmentation_manifest.csv")
    args = parser.parse_args()

    set_global_seed(args.seed)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger("train")

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise SystemExit(f"dataset_dir not found: {dataset_dir}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = dataset_dir / args.manifest
    if manifest_path.is_file():
        samples = load_samples_from_manifest(dataset_dir, manifest_path)
        log.info("loaded samples from manifest: %s | n=%d", manifest_path, len(samples))
    else:
        samples = scan_samples(dataset_dir)
        log.info("scanned samples from folders | n=%d", len(samples))

    if not samples:
        raise SystemExit("no images found")

    groups = {}
    for p, c, g in samples:
        groups.setdefault(g, {"class": c, "paths": []})
        groups[g]["paths"].append(p)

    group_ids = np.array(list(groups.keys()), dtype=object)
    group_classes = np.array([groups[g]["class"] for g in group_ids], dtype=object)

    g_train, g_test = train_test_split(
        group_ids,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
        stratify=group_classes,
    )
    g_train = set(g_train.tolist())
    g_test = set(g_test.tolist())

    train_samples: list[tuple[Path, str]] = []
    test_samples: list[tuple[Path, str]] = []

    for gid, payload in groups.items():
        cls = payload["class"]
        for p in payload["paths"]:
            if gid in g_train:
                train_samples.append((p, cls))
            else:
                test_samples.append((p, cls))

    if not train_samples or not test_samples:
        raise SystemExit("invalid split (empty train or test)")

    log.info(
        "split done | train=%d | test=%d | test_size=%.2f",
        len(train_samples),
        len(test_samples),
        args.test_size,
    )

    img_size = (args.img_size, args.img_size)

    X_train_list: list[np.ndarray] = []
    y_train_list: list[str] = []
    X_test_list: list[np.ndarray] = []
    y_test_list: list[str] = []
    errors: list[tuple[str, str]] = []

    t0 = time.time()
    for p, c in tqdm(train_samples, desc="features_train", unit="img"):
        try:
            X_train_list.append(extract_features_102(p, img_size=img_size))
            y_train_list.append(c)
        except Exception as e:
            errors.append((str(p), str(e)))

    for p, c in tqdm(test_samples, desc="features_test", unit="img"):
        try:
            X_test_list.append(extract_features_102(p, img_size=img_size))
            y_test_list.append(c)
        except Exception as e:
            errors.append((str(p), str(e)))

    if errors:
        err_path = out_dir / "feature_extraction_errors.csv"
        with err_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "error"])
            w.writerows(errors)
        log.info("saved: %s", err_path)

    if not X_train_list or not X_test_list:
        raise SystemExit("no valid samples after feature extraction")

    X_train = np.vstack(X_train_list).astype(np.float64)
    X_test = np.vstack(X_test_list).astype(np.float64)
    y_train = np.array(y_train_list, dtype=object)
    y_test = np.array(y_test_list, dtype=object)

    log.info(
        "features extracted | train=%d | test=%d | failed=%d | elapsed_s=%.3f",
        X_train.shape[0],
        X_test.shape[0],
        len(errors),
        time.time() - t0,
    )

    le = LabelEncoder()
    y_all = np.concatenate([y_train, y_test], axis=0)
    le.fit(y_all)

    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    X_all = np.vstack([X_train, X_test])
    y_all_labels = np.concatenate([y_train, y_test], axis=0)

    df = pd.DataFrame(X_all, columns=[f"f{i:03d}" for i in range(X_all.shape[1])])
    df["label"] = y_all_labels
    features_csv = out_dir / "features_selecionadas.csv"
    df.to_csv(features_csv, index=False, encoding="utf-8")
    log.info("saved: %s", features_csv)

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    specs = build_model_specs(seed=args.seed)

    results = []
    best_estimators: dict[str, object] = {}
    best_params: dict[str, dict] = {}
    test_accs: dict[str, float] = {}
    cv_bests: dict[str, float] = {}

    for spec in specs:
        log.info("grid_search_start: %s", spec.name)
        grid = GridSearchCV(
            estimator=spec.estimator,
            param_grid=spec.param_grid,
            scoring=args.scoring,
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        t1 = time.time()
        grid.fit(X_train, y_train_enc)
        elapsed = time.time() - t1

        y_pred = grid.predict(X_test)
        acc = float(accuracy_score(y_test_enc, y_pred))
        report = classification_report(y_test_enc, y_pred, target_names=le.classes_, digits=4)
        cm = confusion_matrix(y_test_enc, y_pred)

        model_dir = out_dir / "reports" / spec.name
        model_dir.mkdir(parents=True, exist_ok=True)

        write_text(model_dir / "classification_report.txt", report)
        pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_csv(model_dir / "confusion_matrix.csv", encoding="utf-8")
        pd.DataFrame(grid.cv_results_).to_csv(model_dir / "cv_results.csv", index=False, encoding="utf-8")

        results.append(
            {
                "model": spec.name,
                "test_accuracy": acc,
                "cv_best_score": float(grid.best_score_),
                "best_params": grid.best_params_,
                "fit_time_s": float(elapsed),
            }
        )

        best_estimators[spec.name] = grid.best_estimator_
        best_params[spec.name] = grid.best_params_
        test_accs[spec.name] = acc
        cv_bests[spec.name] = float(grid.best_score_)

        log.info(
            "grid_search_done: %s | test_acc=%.4f | cv_best=%.4f | time_s=%.2f",
            spec.name,
            acc,
            grid.best_score_,
            elapsed,
        )

    results_df = pd.DataFrame(results).sort_values(by="test_accuracy", ascending=False)
    (out_dir / "model_comparison.csv").write_text(results_df.to_csv(index=False), encoding="utf-8")
    (out_dir / "model_comparison.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("saved: %s", out_dir / "model_comparison.csv")
    log.info("saved: %s", out_dir / "model_comparison.json")

    if "XGBoost" not in best_estimators:
        raise SystemExit("XGBoost not trained")

    final_model = best_estimators["XGBoost"]

    model_path = out_dir / "modelo_tomate.pkl"
    encoder_path = out_dir / "label_encoder.pkl"
    joblib.dump(final_model, model_path)
    joblib.dump(le, encoder_path)

    meta = {
        "final_model": "XGBoost",
        "final_params": best_params.get("XGBoost", {}),
        "final_test_accuracy": float(test_accs.get("XGBoost", -1.0)),
        "dataset_dir": str(dataset_dir),
        "img_size": [args.img_size, args.img_size],
        "test_size": args.test_size,
        "seed": args.seed,
        "cv_folds": args.cv_folds,
        "scoring": args.scoring,
        "classes": list(le.classes_),
        "feature_vector_size": 102,
        "feature_schema": {
            "haralick_13": 13,
            "zernike_25_degree8": 25,
            "hsv_hist_48": 48,
            "lbp_mean_std_2": 2,
            "hu_7": 7,
            "mean_hsv_3": 3,
            "std_hsv_3": 3,
            "morphology_1": 1,
        },
        "model_summary": {
            "RandomForest": {"test_accuracy": float(test_accs.get("RandomForest", -1.0)), "cv_best": float(cv_bests.get("RandomForest", -1.0))},
            "SVM": {"test_accuracy": float(test_accs.get("SVM", -1.0)), "cv_best": float(cv_bests.get("SVM", -1.0))},
            "XGBoost": {"test_accuracy": float(test_accs.get("XGBoost", -1.0)), "cv_best": float(cv_bests.get("XGBoost", -1.0))},
        },
    }
    (out_dir / "training_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("saved: %s", model_path)
    log.info("saved: %s", encoder_path)
    log.info("saved: %s", out_dir / "training_metadata.json")
    log.info("final_model: XGBoost | test_acc=%.4f", float(test_accs.get("XGBoost", -1.0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
