"""Generate ground truth feature vectors for Rust parity validation.

Usage:
    python scripts/export_feature_vectors.py \
        --images tools/dataset/plantvillage_tomato/raw \
        --out     ml/artifacts/validation \
        --img_size 128 \
        --limit 100

Outputs:
    artifacts/validation/feature_vectors.npy   — shape [N, 188], float64
    artifacts/validation/image_index.json      — {index: relative_path}
    artifacts/validation/zernike_intermediates.npy — shape [N, 25], float64
        (absolute values from mahotas; complex intermediates not exposed by public API)
    artifacts/validation/masks.npy             — shape [N, img_size, img_size], uint8
        (segmented masks — useful to debug zernike/LBP discrepancies)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _collect_images(root: Path, limit: int) -> list[Path]:
    paths = sorted(
        p for p in root.rglob("*") if p.suffix.lower() in ALLOWED_EXT
    )
    if limit > 0:
        paths = paths[:limit]
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description="Export feature vectors ground truth for Rust parity tests.")
    ap.add_argument("--images", required=True, help="Root directory with JPEG images (searched recursively)")
    ap.add_argument("--out", required=True, help="Output directory for .npy and .json files")
    ap.add_argument("--img_size", type=int, default=128, help="Image size used during training (default: 128)")
    ap.add_argument("--limit", type=int, default=0, help="Max images to process (0 = all)")
    args = ap.parse_args()

    images_root = Path(args.images)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not images_root.exists():
        logger.error("images_dir_not_found path={}", images_root)
        return 1

    paths = _collect_images(images_root, args.limit)
    if not paths:
        logger.error("no_images_found root={}", images_root)
        return 1

    logger.info("export_start n_images={} img_size={} out={}", len(paths), args.img_size, out_dir)

    from garden_ml.data.io import decode_bgr_from_bytes, bgr_to_rgb
    from garden_ml.features.extract import ExtractOptions, extract_features_and_meta_from_rgb
    from garden_ml.features.components import zernike_25
    from garden_ml.image.segmentation import segment_leaf_hsv

    opts = ExtractOptions(img_size=args.img_size)

    features_list: list[np.ndarray] = []
    zernike_list: list[np.ndarray] = []
    masks_list: list[np.ndarray] = []
    index: dict[str, str] = {}
    failed = 0

    for i, path in enumerate(tqdm(paths, desc="extracting", unit="img")):
        try:
            data = path.read_bytes()
            bgr = decode_bgr_from_bytes(data)
            rgb = bgr_to_rgb(bgr)

            feat, _meta = extract_features_and_meta_from_rgb(rgb, opts)
            features_list.append(feat)

            # segmented mask for zernike/LBP debug
            _seg, mask = segment_leaf_hsv(rgb, size=(args.img_size, args.img_size))
            masks_list.append(mask)

            # zernike intermediate: mahotas returns |Z_nm|, not complex Z_nm
            # (complex values not exposed by public mahotas API)
            zer = zernike_25(mask)
            zernike_list.append(zer)

            index[str(i)] = str(path.relative_to(images_root) if path.is_relative_to(images_root) else path)

        except Exception as e:
            logger.warning("skip_image path={} error={}", path, e)
            failed += 1

    if not features_list:
        logger.error("all_images_failed")
        return 1

    feat_arr = np.stack(features_list, axis=0)         # [N, 188] float64
    zer_arr = np.stack(zernike_list, axis=0)            # [N, 25]  float64
    mask_arr = np.stack(masks_list, axis=0)             # [N, H, W] uint8

    np.save(out_dir / "feature_vectors.npy", feat_arr)
    np.save(out_dir / "zernike_intermediates.npy", zer_arr)
    np.save(out_dir / "masks.npy", mask_arr)
    (out_dir / "image_index.json").write_text(json.dumps(index, indent=2))

    logger.info(
        "export_done n={} failed={} shape={} out={}",
        len(features_list), failed, feat_arr.shape, out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
