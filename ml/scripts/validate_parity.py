"""Validate feature parity between Python ground truth and Rust server.

Usage:
    python ml/scripts/validate_parity.py \
        --rust-url   http://localhost:5002 \
        --vectors    ml/artifacts/validation/feature_vectors.npy \
        --index      ml/artifacts/validation/image_index.json \
        --images     tools/dataset/plantvillage_tomato/raw \
        --out        ml/artifacts/parity_report.json \
        [--fail-fast] [--limit N]

The script posts each image to Rust /debug/features (requires GML_DEBUG=1),
compares the returned 188-dim vector against the Python ground truth saved by
export_feature_vectors.py, and reports per-component tolerance violations.

Per-component tolerances (absolute):
    [0..13]   haralick_13         atol=1e-7
    [13..38]  zernike_25          atol=1e-8
    [38..86]  hsv_hist_48         atol=1e-10
    [86..145] lbp_nri_uniform_59  atol=1e-10   (should be exact)
    [145..152] hu_7               atol=1e-10
    [152..158] mean_std_hsv_6     atol=1e-10
    [158..164] shape_basic_6      atol=1e-10
    [164..168] lab_ab_mean_std_4  atol=1e-10
    [168..184] lab_ab_hist_16     atol=1e-10
    [184..188] chroma_rg_mean_std atol=1e-10

Note: Rust returns f32 upcast to f64 — absolute difference includes f32 quantisation
(~6 decimal digits). Group A atol=1e-6 is safe; Group B is tighter because those
features span larger value ranges and small relative errors matter more.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import requests

# ── Feature layout ────────────────────────────────────────────────────────────
COMPONENTS = [
    ("haralick_13",        0,   13, 1e-6),
    ("zernike_25",        13,   38, 1e-6),
    ("hsv_hist_48",       38,   86, 1e-6),
    ("lbp_nri_uniform",   86,  145, 1e-6),
    ("hu_7",             145,  152, 1e-6),
    ("mean_std_hsv_6",   152,  158, 1e-6),
    ("shape_basic_6",    158,  164, 1e-6),
    ("lab_ab_mean_std_4",164,  168, 1e-6),
    ("lab_ab_hist_16",   168,  184, 1e-6),
    ("chroma_rg_4",      184,  188, 1e-6),
]


def _query_rust(url: str, image_bytes: bytes, timeout: int = 30) -> np.ndarray:
    """POST image to /debug/features, return 188-dim float64 array."""
    resp = requests.post(
        f"{url}/debug/features",
        data=image_bytes,
        headers={"Content-Type": "image/jpeg"},
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    feats = np.array(payload["features"], dtype=np.float64)
    if feats.shape != (188,):
        raise ValueError(f"Expected 188 features, got {feats.shape}")
    return feats


def _compare(
    idx: int,
    path: str,
    py_feat: np.ndarray,
    rs_feat: np.ndarray,
    fail_fast: bool,
) -> list[dict[str, Any]]:
    """Compare per component; return list of violation dicts (empty = pass)."""
    violations: list[dict[str, Any]] = []

    for name, lo, hi, atol in COMPONENTS:
        py_slice = py_feat[lo:hi]
        rs_slice = rs_feat[lo:hi]
        diff = np.abs(py_slice - rs_slice)
        max_diff = float(diff.max())
        if max_diff > atol:
            worst_i = int(diff.argmax())
            v = {
                "image_idx": idx,
                "image_path": path,
                "component": name,
                "feat_indices": f"[{lo}:{hi}]",
                "max_abs_diff": max_diff,
                "atol": atol,
                "worst_feat_idx": lo + worst_i,
                "py_value": float(py_slice[worst_i]),
                "rs_value": float(rs_slice[worst_i]),
            }
            violations.append(v)
            if fail_fast:
                _print_violation(v)
                raise SystemExit(1)
    return violations


def _print_violation(v: dict[str, Any]) -> None:
    print(
        f"  FAIL  component={v['component']}  "
        f"feat={v['worst_feat_idx']}  "
        f"max_diff={v['max_abs_diff']:.3e}  "
        f"(atol={v['atol']:.0e})  "
        f"py={v['py_value']:.9f}  rs={v['rs_value']:.9f}  "
        f"image={v['image_path']}",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate Rust feature parity against Python ground truth.")
    ap.add_argument("--rust-url",  default="http://localhost:5002", help="Rust server base URL")
    ap.add_argument("--vectors",   required=True, help="Path to feature_vectors.npy")
    ap.add_argument("--index",     required=True, help="Path to image_index.json")
    ap.add_argument("--images",    required=True, help="Root directory of images (used to build full paths)")
    ap.add_argument("--out",       default=None,  help="Write parity_report.json here (optional)")
    ap.add_argument("--fail-fast", action="store_true", help="Stop on first violation and print diff")
    ap.add_argument("--limit",     type=int, default=0, help="Limit to first N images (0 = all)")
    ap.add_argument("--timeout",   type=int, default=30, help="HTTP timeout per request in seconds")
    args = ap.parse_args()

    vectors_path = Path(args.vectors)
    index_path = Path(args.index)
    images_root = Path(args.images)

    if not vectors_path.exists():
        print(f"ERROR: feature_vectors.npy not found: {vectors_path}", file=sys.stderr)
        print("Run export_feature_vectors.py first.", file=sys.stderr)
        return 1
    if not index_path.exists():
        print(f"ERROR: image_index.json not found: {index_path}", file=sys.stderr)
        return 1

    gt = np.load(vectors_path)             # [N, 188] float64
    index: dict[str, str] = json.loads(index_path.read_text())

    n_total = gt.shape[0]
    if args.limit > 0:
        n_total = min(n_total, args.limit)

    print(f"Validating {n_total} images against {args.rust_url} …")
    print(f"  ground truth: {vectors_path}  shape={gt.shape}")

    all_violations: list[dict[str, Any]] = []
    n_pass = 0
    n_fail = 0
    n_error = 0

    for i in range(n_total):
        rel_path = index.get(str(i), "")
        full_path = images_root / rel_path

        if not full_path.exists():
            print(f"  SKIP  [{i}] not found: {full_path}", flush=True)
            n_error += 1
            continue

        try:
            image_bytes = full_path.read_bytes()
            rs_feat = _query_rust(args.rust_url, image_bytes, args.timeout)
        except Exception as e:
            print(f"  ERROR [{i}] {rel_path}: {e}", flush=True)
            n_error += 1
            continue

        py_feat = gt[i]  # float64 ground truth
        violations = _compare(i, rel_path, py_feat, rs_feat, args.fail_fast)

        if violations:
            n_fail += 1
            all_violations.extend(violations)
            if not args.fail_fast:
                for v in violations:
                    _print_violation(v)
        else:
            n_pass += 1
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{n_total}] {n_pass} pass, {n_fail} fail, {n_error} error", flush=True)

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print(f"Results: {n_pass} pass / {n_fail} fail / {n_error} error  (total={n_total})")

    # Per-component summary
    if all_violations:
        comp_stats: dict[str, list[float]] = {}
        for v in all_violations:
            comp_stats.setdefault(v["component"], []).append(v["max_abs_diff"])
        print("\nPer-component failure summary:")
        for name, diffs in sorted(comp_stats.items()):
            print(f"  {name:22s}  n_fail={len(diffs):4d}  max_diff={max(diffs):.3e}")

    if args.out:
        report = {
            "n_total": n_total,
            "n_pass": n_pass,
            "n_fail": n_fail,
            "n_error": n_error,
            "pass_rate": n_pass / max(n_pass + n_fail, 1),
            "violations": all_violations,
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\nReport written to {out_path}")

    if n_fail > 0:
        print(f"\nPARITY FAILED — {n_fail} images have feature divergences above tolerance.")
        return 1

    print("\nPARITY OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
