import argparse
import csv
import os
import random
import zlib
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


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


def stable_int_seed(text: str) -> int:
    return int(zlib.adler32(text.encode("utf-8")) & 0xFFFFFFFF)


def build_datagen() -> ImageDataGenerator:
    def aug_preprocess(img: np.ndarray) -> np.ndarray:
        arr = np.clip(img, 0, 255).astype(np.uint8)

        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        delta = np.random.randint(-5, 6)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + delta) % 180
        arr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        gamma = float(np.random.uniform(0.9, 1.1))
        inv = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv * 255 for i in range(256)], dtype=np.float32)
        arr = cv2.LUT(arr, table.astype(np.uint8))

        return arr.astype(np.float32)

    return ImageDataGenerator(
        rotation_range=25,
        brightness_range=(0.7, 1.3),
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        preprocessing_function=aug_preprocess,
    )


def save_jpeg(rgb: np.ndarray, out_path: Path, quality: int = 95) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out_path, format="JPEG", quality=quality, optimize=True)


def relpath_str(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="dataset")
    parser.add_argument("--output_dir", type=str, default="dataset_aug")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--aug_per_image", type=int, default=5)
    parser.add_argument("--no_originals", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--manifest", type=str, default="augmentation_manifest.csv")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    img_size = (args.img_size, args.img_size)

    if not in_dir.exists() or not in_dir.is_dir():
        raise SystemExit(f"input_dir not found: {in_dir}")

    set_global_seed(args.seed)
    datagen = build_datagen()

    classes = sorted([p.name for p in in_dir.iterdir() if p.is_dir()])
    if not classes:
        raise SystemExit(f"no class folders found inside: {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    items: list[tuple[str, Path]] = []
    for c in classes:
        class_in = in_dir / c
        for p in sorted([x for x in class_in.iterdir() if x.is_file() and is_image_file(x)]):
            items.append((c, p))

    if not items:
        raise SystemExit(f"no images found inside: {in_dir}")

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = out_dir / manifest_path

    rows: list[list[str]] = []
    pbar = tqdm(total=len(items), desc="augmenting", unit="img")

    include_originals = not args.no_originals

    for c, img_path in items:
        pbar.update(1)

        group_id = f"{c}/{img_path.stem}"
        class_out = out_dir / c
        class_out.mkdir(parents=True, exist_ok=True)

        try:
            pil = Image.open(img_path).convert("RGB")
            rgb = np.array(pil)
            rgb = letterbox_rgb(rgb, img_size, bg=0)

            src_rel = relpath_str(img_path, in_dir)

            seed_img = (args.seed + stable_int_seed(f"{src_rel}|{group_id}")) % (2**31 - 1)

            if include_originals:
                out_orig = class_out / f"{img_path.stem}__orig.jpg"
                save_jpeg(rgb, out_orig, quality=args.jpeg_quality)
                rows.append(
                    [
                        c,
                        group_id,
                        src_rel,
                        relpath_str(out_orig, out_dir),
                        "orig",
                        "",
                        str(seed_img),
                        "ok",
                        "",
                    ]
                )

            x = rgb.astype(np.float32)
            x = np.expand_dims(x, axis=0)

            flow = datagen.flow(x, batch_size=1, shuffle=False, seed=seed_img)

            for k in range(1, args.aug_per_image + 1):
                batch = next(flow)[0]
                aug = np.clip(batch, 0, 255).astype(np.uint8)
                out_aug = class_out / f"{img_path.stem}__aug{k:03d}.jpg"
                save_jpeg(aug, out_aug, quality=args.jpeg_quality)
                rows.append(
                    [
                        c,
                        group_id,
                        src_rel,
                        relpath_str(out_aug, out_dir),
                        "aug",
                        str(k),
                        str(seed_img),
                        "ok",
                        "",
                    ]
                )

        except Exception as e:
            rows.append(
                [
                    c,
                    group_id,
                    relpath_str(img_path, in_dir),
                    "",
                    "error",
                    "",
                    str(args.seed),
                    "error",
                    str(e).replace("\n", " ").strip(),
                ]
            )

    pbar.close()

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "group_id", "source_path", "output_path", "kind", "aug_index", "seed", "status", "error"])
        w.writerows(rows)

    print(f"done: {out_dir}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
