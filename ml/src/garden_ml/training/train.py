from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
from xgboost import XGBClassifier

from garden_ml.config.constants import ENCODER_FILE, FEATURE_SCHEMA_FILE, MODEL_FILE, TRAIN_META_FILE
from garden_ml.config.logging import setup_logging
from garden_ml.config.settings import Settings
from garden_ml.data.manifest import class_distribution, samples_from_manifest, scan_folder_dataset, write_json
from garden_ml.data.splits import expand_groups, stratified_group_split
from garden_ml.features.extract import ExtractOptions, extract_features_from_path
from garden_ml.features.schema import SCHEMA, write_schema


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Any
    param_grid: dict[str, list[Any]]


def build_specs(seed: int) -> list[ModelSpec]:
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
                "n_estimators": [200, 400],
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
                "svc__C": [0.1, 1.0, 10.0],
                "svc__gamma": ["scale", "auto"],
            },
        ),
        ModelSpec(
            name="XGBoost",
            estimator=xgb,
            param_grid={
                "n_estimators": [200, 400],
                "max_depth": [3, 6, 10],
                "learning_rate": [0.05, 0.10],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        ),
    ]


def group_cv(n_splits: int, seed: int):
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    except Exception:
        from sklearn.model_selection import GroupKFold

        return GroupKFold(n_splits=n_splits)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    bal = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=np.arange(len(labels)), zero_division=0)
    macro_f1 = float(np.mean(f1))
    weighted_f1 = float(np.average(f1, weights=sup)) if int(sup.sum()) > 0 else 0.0

    per_class = []
    for i, name in enumerate(labels):
        per_class.append(
            {
                "class": name,
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(sup[i]),
            }
        )

    return {
        "accuracy": acc,
        "balanced_accuracy": bal,
        "mcc": mcc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
    }


def maybe_mlflow_init(st: Settings) -> None:
    if not st.mlflow_enabled:
        return
    import mlflow

    mlflow.set_tracking_uri(st.mlflow_tracking_uri)
    mlflow.set_experiment(st.mlflow_experiment)


def maybe_mlflow_log(params: dict[str, Any], metrics: dict[str, Any], artifacts: dict[str, Path], tags: dict[str, str], enabled: bool) -> None:
    if not enabled:
        return
    import mlflow

    with mlflow.start_run():
        mlflow.set_tags(tags)
        mlflow.log_params(params)
        flat = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(flat)
        for name, p in artifacts.items():
            if p.is_file():
                mlflow.log_artifact(str(p), artifact_path=name)


def _infer_kind_from_path(p: str) -> str:
    stem = Path(p).stem.lower()
    if "__aug" in stem:
        return "aug"
    if "__orig" in stem:
        return "orig"
    return "orig"


def _write_manifest_csv(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "class", "group_id", "kind"])
        for r in rows:
            w.writerow(list(r))


def _kind_counts(rows: list[tuple[str, str, str, str]]) -> dict[str, int]:
    d: dict[str, int] = {}
    for _p, _c, _g, k in rows:
        d[k] = d.get(k, 0) + 1
    return dict(sorted(d.items(), key=lambda x: x[0]))


def _md5_file(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, default="dataset_aug")
    ap.add_argument("--output_dir", type=str, default="artifacts/model_registry/v0004")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--test_size", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--manifest", type=str, default="augmentation_manifest.csv")
    ap.add_argument("--photometric_normalize", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    st = Settings(img_size=int(args.img_size), seed=int(args.seed), artifacts_dir=out_dir)
    setup_logging(st.log_level)

    st.ensure_dirs()
    set_seed(int(args.seed))
    maybe_mlflow_init(st)

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        raise SystemExit(f"dataset_dir not found: {dataset_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = dataset_dir / str(args.manifest)
    from_manifest = manifest_path.is_file()

    logger.info(
        "train_start dataset_dir={} output_dir={} img_size={} test_size={} seed={} cv_folds={} manifest={} photometric_normalize={} mlflow_enabled={}",
        str(dataset_dir),
        str(out_dir),
        int(args.img_size),
        float(args.test_size),
        int(args.seed),
        int(args.cv_folds),
        str(args.manifest),
        bool(args.photometric_normalize),
        bool(st.mlflow_enabled),
    )

    try:
        samples = samples_from_manifest(dataset_dir, args.manifest, include_kinds={"orig", "aug"})
        logger.info("loaded_from_manifest n={}", len(samples))
    except Exception as e:
        logger.warning("manifest_load_failed {}", str(e))
        raw = scan_folder_dataset(dataset_dir)
        samples = [(p, c, g, "orig") for p, c, g in raw]
        logger.info("scanned_folders n={}", len(samples))

    if not samples:
        raise SystemExit("no samples found")

    all_kind_counts = {"orig": 0, "aug": 0}
    for _p, _c, _g, k in samples:
        if k in all_kind_counts:
            all_kind_counts[k] += 1
    all_kind_counts = dict(sorted(all_kind_counts.items(), key=lambda x: x[0]))

    if from_manifest and int(all_kind_counts.get("aug", 0)) == 0:
        raise SystemExit("manifest exists but no aug samples were found")

    orig_md5_by_path: dict[str, str] = {}
    md5_to_classes: dict[str, set[str]] = {}
    md5_to_gids: dict[str, set[str]] = {}

    first_gid_by_key: dict[tuple[str, str], str] = {}
    gid_remap: dict[str, str] = {}

    orig_only = [(p, cls, gid, kind) for (p, cls, gid, kind) in samples if str(kind) == "orig"]
    for p, cls, gid, _kind in tqdm(orig_only, desc="hash_orig", unit="img", mininterval=1.0):
        hp = _md5_file(Path(p))
        orig_md5_by_path[str(p)] = hp
        md5_to_classes.setdefault(hp, set()).add(str(cls))
        md5_to_gids.setdefault(hp, set()).add(str(gid))

        key = (str(cls), hp)
        if key in first_gid_by_key:
            tgt = first_gid_by_key[key]
            if str(gid) != tgt:
                gid_remap[str(gid)] = tgt
        else:
            first_gid_by_key[key] = str(gid)

    bad_md5 = sorted([h for h, cs in md5_to_classes.items() if len(cs) > 1])
    bad_gids: set[str] = set()
    for h in bad_md5:
        bad_gids.update(md5_to_gids.get(h, set()))

    if bad_gids:
        samples = [(p, cls, gid, kind) for (p, cls, gid, kind) in samples if str(gid) not in bad_gids]

    if gid_remap:
        samples = [(p, cls, gid_remap.get(str(gid), str(gid)), kind) for (p, cls, gid, kind) in samples]

    integrity = {
        "cross_class_duplicate_hashes": int(len(bad_md5)),
        "cross_class_duplicate_groups_removed": int(len(bad_gids)),
        "dedup_gid_remap_count": int(len(gid_remap)),
    }
    (out_dir / "dataset_integrity.json").write_text(json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8")

    groups: dict[str, dict[str, Any]] = {}
    for path, cls, gid, kind in samples:
        groups.setdefault(str(gid), {"class": str(cls), "items": []})
        groups[str(gid)]["items"].append((str(path), str(kind)))

    group_ids = list(groups.keys())
    group_labels = [groups[g]["class"] for g in group_ids]
    split = stratified_group_split(group_ids, group_labels, test_size=float(args.test_size), seed=int(args.seed))

    overlap_groups = split.train_groups.intersection(split.test_groups)
    if overlap_groups:
        raise SystemExit(f"group leakage detected: overlap_groups={len(overlap_groups)}")

    expanded_samples: list[tuple[str, str, str, str]] = []
    for gid, payload in groups.items():
        cls = payload["class"]
        for path, kind in payload["items"]:
            expanded_samples.append((str(path), str(cls), str(gid), str(kind)))

    train_items, test_items, train_g, test_g = expand_groups(
        expanded_samples, split.train_groups, split.test_groups, test_kinds_only={"orig"}
    )
    if not train_items or not test_items:
        raise SystemExit("invalid split (empty train or test)")

    train_manifest_rows: list[tuple[str, str, str, str]] = []
    for (p, c), gid in zip(train_items, train_g):
        train_manifest_rows.append((str(p), str(c), str(gid), _infer_kind_from_path(str(p))))
    train_kind_counts = _kind_counts(train_manifest_rows)

    if from_manifest and int(all_kind_counts.get("aug", 0)) > 0 and int(train_kind_counts.get("aug", 0)) == 0:
        raise SystemExit("aug samples exist in dataset, but none were assigned to training set")

    train_orig_hash: set[str] = set()
    for (p, _c), _gid in zip(train_items, train_g):
        if _infer_kind_from_path(str(p)) == "orig":
            h = orig_md5_by_path.get(str(p))
            if h is None:
                h = _md5_file(Path(p))
                orig_md5_by_path[str(p)] = h
            train_orig_hash.add(h)

    dup_found: list[dict[str, str]] = []
    kept_test_items: list[tuple[str, str]] = []
    kept_test_g: list[str] = []

    for (p, c), gid in zip(test_items, test_g):
        h = orig_md5_by_path.get(str(p))
        if h is None:
            h = _md5_file(Path(p))
            orig_md5_by_path[str(p)] = h
        if h in train_orig_hash:
            dup_found.append({"hash": h, "test_path": str(p)})
            continue
        kept_test_items.append((str(p), str(c)))
        kept_test_g.append(str(gid))

    test_items = kept_test_items
    test_g = kept_test_g
    if not test_items:
        raise SystemExit("all test samples removed due to hash overlap with train")

    test_manifest_rows: list[tuple[str, str, str, str]] = []
    for (p, c), gid in zip(test_items, test_g):
        test_manifest_rows.append((str(p), str(c), str(gid), "orig"))
    test_kind_counts = _kind_counts(test_manifest_rows)

    _write_manifest_csv(out_dir / "train_manifest.csv", train_manifest_rows)
    _write_manifest_csv(out_dir / "test_manifest.csv", test_manifest_rows)

    split_payload = {
        "seed": int(args.seed),
        "test_size": float(args.test_size),
        "train_groups": sorted(list(split.train_groups)),
        "test_groups": sorted(list(split.test_groups)),
        "n_train_samples": int(len(train_items)),
        "n_test_samples": int(len(test_items)),
        "kind_counts_all": all_kind_counts,
        "train_kind_counts": train_kind_counts,
        "test_kind_counts": test_kind_counts,
        "hash_overlap_removed_from_test": int(len(dup_found)),
        "grid_n_jobs": int(st.grid_n_jobs),
        "grid_pre_dispatch": str(st.grid_pre_dispatch),
    }
    (out_dir / "split_groups.json").write_text(json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    leakage_report = {
        "groups_overlap": 0,
        "train_kind_counts": train_kind_counts,
        "test_kind_counts": test_kind_counts,
        "orig_hash_overlap_removed_from_test": int(len(dup_found)),
        "orig_hash_overlap_examples": dup_found[:50],
    }
    (out_dir / "leakage_check.json").write_text(json.dumps(leakage_report, ensure_ascii=False, indent=2), encoding="utf-8")

    opts = ExtractOptions(img_size=int(args.img_size), photometric_normalize=bool(args.photometric_normalize))

    X_train_list: list[np.ndarray] = []
    y_train_list: list[str] = []
    g_train_list: list[str] = []
    X_test_list: list[np.ndarray] = []
    y_test_list: list[str] = []
    errors: list[dict[str, str]] = []

    t0 = time.time()
    for ((p, c), gid) in tqdm(list(zip(train_items, train_g)), desc="features_train", unit="img", mininterval=1.0):
        try:
            X_train_list.append(extract_features_from_path(Path(p), opts))
            y_train_list.append(str(c))
            g_train_list.append(str(gid))
        except Exception as e:
            errors.append({"path": str(p), "error": str(e)})

    for p, c in tqdm(test_items, desc="features_test", unit="img", mininterval=1.0):
        try:
            X_test_list.append(extract_features_from_path(Path(p), opts))
            y_test_list.append(str(c))
        except Exception as e:
            errors.append({"path": str(p), "error": str(e)})

    if not X_train_list or not X_test_list:
        raise SystemExit("no valid samples after feature extraction")

    X_train = np.vstack(X_train_list).astype(np.float32, copy=False)
    X_test = np.vstack(X_test_list).astype(np.float32, copy=False)
    y_train = np.array(y_train_list, dtype=object)
    y_test = np.array(y_test_list, dtype=object)
    g_train_arr = np.array(g_train_list, dtype=object)

    logger.info(
        "features_done train={} test={} failed={} elapsed_s={:.3f}",
        int(X_train.shape[0]),
        int(X_test.shape[0]),
        int(len(errors)),
        float(time.time() - t0),
    )

    le = LabelEncoder()
    le.fit(y_train)

    unseen = sorted(list(set(np.unique(y_test)).difference(set(le.classes_))))
    if unseen:
        raise SystemExit(f"test has unseen classes (label leakage/stratify issue): {unseen}")

    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    write_schema(out_dir / FEATURE_SCHEMA_FILE)

    if bool(st.dump_features_csv):
        df_all = pd.DataFrame(
            np.vstack([X_train, X_test]),
            columns=[f"f{i:03d}" for i in range(int(X_train.shape[1]))],
        )
        df_all["label"] = np.concatenate([y_train, y_test], axis=0)
        df_all.to_csv(out_dir / "features_188.csv", index=False, encoding="utf-8")
        del df_all
        gc.collect()

    cv = group_cv(n_splits=int(args.cv_folds), seed=int(args.seed))
    specs = build_specs(seed=int(args.seed))

    best_name = None
    best_est = None
    best_score = -1.0
    leaderboard: list[dict[str, Any]] = []
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        logger.info("grid_search_start model={} grid_n_jobs={} pre_dispatch={}", spec.name, int(st.grid_n_jobs), str(st.grid_pre_dispatch))

        grid = GridSearchCV(
            estimator=spec.estimator,
            param_grid=spec.param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=int(st.grid_n_jobs),
            pre_dispatch=str(st.grid_pre_dispatch),
            refit=True,
            verbose=1,
            error_score="raise",
        )

        t1 = time.time()
        grid.fit(X_train, y_train_enc, groups=g_train_arr)
        fit_s = float(time.time() - t1)

        y_pred = grid.predict(X_test)
        metrics = compute_metrics(y_test_enc, y_pred, labels=list(le.classes_))
        cm = confusion_matrix(y_test_enc, y_pred)

        model_dir = reports_dir / spec.name
        model_dir.mkdir(parents=True, exist_ok=True)

        (model_dir / "classification_report.txt").write_text(
            classification_report(y_test_enc, y_pred, target_names=le.classes_, digits=4),
            encoding="utf-8",
        )
        pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_csv(model_dir / "confusion_matrix.csv", encoding="utf-8")
        pd.DataFrame(grid.cv_results_).to_csv(model_dir / "cv_results.csv", index=False, encoding="utf-8")

        entry = {
            "model": spec.name,
            "cv_best_f1_macro": float(grid.best_score_),
            "test_accuracy": metrics["accuracy"],
            "test_macro_f1": metrics["macro_f1"],
            "test_weighted_f1": metrics["weighted_f1"],
            "test_balanced_accuracy": metrics["balanced_accuracy"],
            "test_mcc": metrics["mcc"],
            "best_params": dict(grid.best_params_),
            "fit_time_s": fit_s,
        }
        leaderboard.append(entry)

        if float(metrics["macro_f1"]) > float(best_score):
            best_score = float(metrics["macro_f1"])
            best_name = str(spec.name)
            best_est = grid.best_estimator_

        logger.info(
            "grid_search_done model={} test_macro_f1={:.4f} test_acc={:.4f} cv_best={:.4f} time_s={:.2f}",
            spec.name,
            float(metrics["macro_f1"]),
            float(metrics["accuracy"]),
            float(grid.best_score_),
            float(fit_s),
        )

        gc.collect()

    pd.DataFrame(leaderboard).sort_values(by="test_macro_f1", ascending=False).to_csv(out_dir / "model_comparison.csv", index=False)
    write_json(out_dir / "model_comparison.json", {"leaderboard": leaderboard})

    if best_est is None or best_name is None:
        raise SystemExit("no model selected")

    joblib.dump(best_est, out_dir / MODEL_FILE)
    joblib.dump(le, out_dir / ENCODER_FILE)

    meta = {
        "final_model": best_name,
        "final_macro_f1": float(best_score),
        "img_size": int(args.img_size),
        "test_size": float(args.test_size),
        "seed": int(args.seed),
        "cv_folds": int(args.cv_folds),
        "classes": list(le.classes_),
        "feature_schema": SCHEMA.as_dict(),
        "feature_schema_sha1": SCHEMA.sha1(),
        "photometric_normalize": bool(args.photometric_normalize),
        "dataset_dir": str(dataset_dir),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "features_dim": int(X_train.shape[1]),
        "class_distribution_all": class_distribution([(cls, "") for cls in np.concatenate([y_train, y_test], axis=0)]),
        "class_distribution_train": class_distribution([(cls, "") for cls in y_train]),
        "class_distribution_test": class_distribution([(cls, "") for cls in y_test]),
        "errors_count": int(len(errors)),
        "kind_counts_all": all_kind_counts,
        "train_kind_counts": train_kind_counts,
        "test_kind_counts": test_kind_counts,
        "leakage_check": {"groups_overlap": 0, "orig_hash_overlap_removed_from_test": int(len(dup_found))},
        "dataset_integrity": integrity,
        "grid_n_jobs": int(st.grid_n_jobs),
        "grid_pre_dispatch": str(st.grid_pre_dispatch),
        "dump_features_csv": bool(st.dump_features_csv),
    }
    (out_dir / TRAIN_META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        (out_dir / "feature_extraction_errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")

    artifacts = {
        "model": out_dir / MODEL_FILE,
        "encoder": out_dir / ENCODER_FILE,
        "meta": out_dir / TRAIN_META_FILE,
        "schema": out_dir / FEATURE_SCHEMA_FILE,
        "comparison": out_dir / "model_comparison.csv",
        "split_groups": out_dir / "split_groups.json",
        "train_manifest": out_dir / "train_manifest.csv",
        "test_manifest": out_dir / "test_manifest.csv",
        "leakage_check": out_dir / "leakage_check.json",
        "dataset_integrity": out_dir / "dataset_integrity.json",
        "features": out_dir / "features_188.csv",
    }
    params = {
        "seed": int(args.seed),
        "img_size": int(args.img_size),
        "test_size": float(args.test_size),
        "cv_folds": int(args.cv_folds),
        "final_model": best_name,
        "photometric_normalize": bool(args.photometric_normalize),
        "features_dim": int(X_train.shape[1]),
        "feature_schema_sha1": SCHEMA.sha1(),
        "grid_n_jobs": int(st.grid_n_jobs),
        "grid_pre_dispatch": str(st.grid_pre_dispatch),
        "dump_features_csv": bool(st.dump_features_csv),
    }
    tags = {"pipeline": "features_188", "selection": "best_macro_f1", "final_model": best_name}
    maybe_mlflow_log(params=params, metrics={"final_macro_f1": float(best_score)}, artifacts=artifacts, tags=tags, enabled=st.mlflow_enabled)

    logger.info("train_done artifacts_dir={} final_model={} macro_f1={:.4f} errors_count={}", str(out_dir), best_name, float(best_score), len(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
