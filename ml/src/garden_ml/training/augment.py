from __future__ import annotations

import argparse
import csv
import json
import os
import random
import zlib
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from garden_ml.config.constants import DEFAULT_AUG_CONFIG, DEFAULT_AUG_MANIFEST
from garden_ml.config.logging import setup_logging
from garden_ml.data.io import is_image_file
from garden_ml.image.resize import letterbox_rgb
from garden_ml.image.segmentation import segment_leaf_hsv

from loguru import logger


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def stable_int_seed(text: str) -> int:
    return int(zlib.adler32(text.encode("utf-8")) & 0xFFFFFFFF)


def save_jpeg(rgb: np.ndarray, out_path: Path, quality: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out_path, format="JPEG", quality=int(quality), optimize=True)


def build_aug_pipeline() -> tuple[A.Compose, dict]:
    pipeline = A.Compose(
        [
            A.OneOf(
                [
                    A.Rotate(limit=25, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
                    A.ShiftScaleRotate(
                        shift_limit=0.20,
                        scale_limit=0.20,
                        rotate_limit=20,
                        border_mode=cv2.BORDER_REFLECT_101,
                        p=1.0,
                    ),
                ],
                p=0.80,
            ),
            A.HorizontalFlip(p=0.50),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.30, contrast_limit=0.25, p=1.0),
                    A.RandomGamma(gamma_limit=(90, 110), p=1.0),
                    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=10, p=1.0),
                ],
                p=0.85,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ],
                p=0.25,
            ),
        ],
        p=1.0,
    )

    desc = {
        "rotation_limit_deg": 25,
        "shift_limit_frac": 0.20,
        "scale_limit_frac": 0.20,
        "brightness_limit": 0.30,
        "contrast_limit": 0.25,
        "gamma_range": [0.90, 1.10],
        "hue_shift_deg": 5,
        "sat_shift": 15,
        "val_shift": 10,
        "blur_kernel": [3, 5],
        "median_blur": 3,
    }
    return pipeline, desc


def main() -> int:
    setup_logging()

    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, default="dataset")
    p.add_argument("--output_dir", type=str, default="dataset_aug")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--aug_per_image", type=int, default=5)
    p.add_argument("--no_originals", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--jpeg_quality", type=int, default=95)
    p.add_argument("--manifest", type=str, default=DEFAULT_AUG_MANIFEST)
    p.add_argument("--segment_before_aug", action="store_true")
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    if not in_dir.is_dir():
        raise SystemExit(f"input_dir not found: {in_dir}")

    set_seed(int(args.seed))
    aug, aug_desc = build_aug_pipeline()

    classes = sorted([x.name for x in in_dir.iterdir() if x.is_dir()])
    if not classes:
        raise SystemExit(f"no class folders found: {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    items: list[tuple[str, Path]] = []
    for c in classes:
        for f in sorted([x for x in (in_dir / c).iterdir() if x.is_file() and is_image_file(x)]):
            items.append((c, f))
    if not items:
        raise SystemExit(f"no images in: {in_dir}")

    manifest_path = out_dir / args.manifest
    config_path = out_dir / DEFAULT_AUG_CONFIG

    config_payload = {
        "seed": int(args.seed),
        "img_size": int(args.img_size),
        "aug_per_image": int(args.aug_per_image),
        "jpeg_quality": int(args.jpeg_quality),
        "no_originals": bool(args.no_originals),
        "segment_before_aug": bool(args.segment_before_aug),
        "pipeline": aug_desc,
    }
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "augment_start input={} output={} classes={} images={} img_size={} aug_per_image={} seed={} segment_before_aug={}",
        str(in_dir),
        str(out_dir),
        len(classes),
        len(items),
        int(args.img_size),
        int(args.aug_per_image),
        int(args.seed),
        bool(args.segment_before_aug),
    )

    rows: list[list[str]] = []
    include_originals = not bool(args.no_originals)
    size = (int(args.img_size), int(args.img_size))

    saved_orig = 0
    saved_aug = 0
    failed = 0

    bar = tqdm(total=len(items), desc="augmenting", unit="img", mininterval=1.0)
    for c, img_path in items:
        bar.update(1)
        group_id = f"{c}/{img_path.stem}"
        class_out = out_dir / c
        class_out.mkdir(parents=True, exist_ok=True)

        pil = None
        rgb = None
        try:
            pil = Image.open(img_path).convert("RGB")
            rgb = np.asarray(pil, dtype=np.uint8)
            rgb = letterbox_rgb(rgb, size=size, bg=0)

            if bool(args.segment_before_aug):
                rgb, _mask = segment_leaf_hsv(rgb, size=size)

            src_rel = str(img_path.resolve().relative_to(in_dir.resolve()))
            seed_img = (int(args.seed) + stable_int_seed(f"{src_rel}|{group_id}")) % (2**31 - 1)

            if include_originals:
                out_orig = class_out / f"{img_path.stem}__orig.jpg"
                save_jpeg(rgb, out_orig, quality=int(args.jpeg_quality))
                rows.append([c, group_id, src_rel, str(out_orig.relative_to(out_dir)), "orig", "", str(seed_img), "ok", ""])
                saved_orig += 1

            for k in range(1, int(args.aug_per_image) + 1):
                rng = np.random.default_rng(seed_img + k)
                A.set_seed(int(rng.integers(0, 2**31 - 1)))
                auged = aug(image=rgb)["image"]
                auged = np.clip(auged, 0, 255).astype(np.uint8)
                out_aug = class_out / f"{img_path.stem}__aug{k:03d}.jpg"
                save_jpeg(auged, out_aug, quality=int(args.jpeg_quality))
                rows.append([c, group_id, src_rel, str(out_aug.relative_to(out_dir)), "aug", str(k), str(seed_img), "ok", ""])
                saved_aug += 1

        except Exception as e:
            failed += 1
            rows.append([c, group_id, str(img_path), "", "error", "", str(int(args.seed)), "error", str(e).replace("\n", " ").strip()])

    bar.close()

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "group_id", "source_path", "output_path", "kind", "aug_index", "seed", "status", "error"])
        w.writerows(rows)

    if int(args.aug_per_image) > 0 and saved_aug == 0:
        raise SystemExit("augmentation produced zero augmented samples (check pipeline and output dir)")

    logger.info(
        "augment_done output={} manifest={} config={} saved_orig={} saved_aug={} failed={}",
        str(out_dir),
        str(manifest_path),
        str(config_path),
        int(saved_orig),
        int(saved_aug),
        int(failed),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
