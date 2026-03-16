from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, precision_recall_fscore_support
from tqdm import tqdm

from garden_ml.config.logging import setup_logging
from garden_ml.data.manifest import samples_from_manifest, scan_folder_dataset
from garden_ml.data.splits import stratified_group_split

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
except Exception:
    torch = None


@dataclass(frozen=True)
class Sample:
    path: Path
    cls: str
    gid: str
    kind: str


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _md5_file(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class ImgDataset(Dataset):
    def __init__(self, items: list[Sample], class_to_idx: dict[str, int], tfm):
        self.items = items
        self.class_to_idx = class_to_idx
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        from PIL import Image

        s = self.items[i]
        img = Image.open(s.path).convert("RGB")
        x = self.tfm(img)
        y = self.class_to_idx[s.cls]
        return x, y


def metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    bal = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=np.arange(n_classes), zero_division=0)
    macro_f1 = float(np.mean(f1))
    weighted_f1 = float(np.average(f1, weights=sup)) if sup.sum() > 0 else 0.0
    return {"accuracy": acc, "balanced_accuracy": bal, "mcc": mcc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}


def main() -> int:
    setup_logging()

    if torch is None:
        raise SystemExit("install extras: pip install -e '.[dl]'")

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, default="dataset_aug")
    ap.add_argument("--output_dir", type=str, default="artifacts/model_registry/v0004_dl")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.30)
    ap.add_argument("--manifest", type=str, default="augmentation_manifest.csv")
    args = ap.parse_args()

    set_seed(int(args.seed))

    # Maximize CPU thread usage for PyTorch math operations
    n_cpu = os.cpu_count() or 1
    torch.set_num_threads(n_cpu)
    torch.set_num_interop_threads(max(1, n_cpu // 2))

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "deep_start dataset_dir={} output_dir={} img_size={} epochs={} batch={} lr={} seed={} test_size={}",
        str(dataset_dir),
        str(out_dir),
        int(args.img_size),
        int(args.epochs),
        int(args.batch),
        float(args.lr),
        int(args.seed),
        float(args.test_size),
    )

    try:
        rows = samples_from_manifest(dataset_dir, args.manifest, include_kinds={"orig", "aug"}, require_status_ok=True)
        samples = [Sample(Path(p), cls, gid, kind) for p, cls, gid, kind in rows]
    except Exception:
        raw = scan_folder_dataset(dataset_dir)
        samples = [Sample(p, cls, gid, "orig") for p, cls, gid in raw]

    orig_md5_by_gid: dict[str, str] = {}
    md5_to_classes: dict[str, set[str]] = {}
    md5_to_gids: dict[str, set[str]] = {}
    first_gid_by_key: dict[tuple[str, str], str] = {}
    gid_remap: dict[str, str] = {}

    for s in tqdm([x for x in samples if x.kind == "orig"], desc="deep_hash_orig", unit="img", mininterval=1.0):
        h = _md5_file(s.path)
        orig_md5_by_gid[str(s.gid)] = h
        md5_to_classes.setdefault(h, set()).add(str(s.cls))
        md5_to_gids.setdefault(h, set()).add(str(s.gid))

        key = (str(s.cls), h)
        if key in first_gid_by_key:
            tgt = first_gid_by_key[key]
            if str(s.gid) != tgt:
                gid_remap[str(s.gid)] = tgt
        else:
            first_gid_by_key[key] = str(s.gid)

    bad_md5 = [h for h, cs in md5_to_classes.items() if len(cs) > 1]
    bad_gids: set[str] = set()
    for h in bad_md5:
        bad_gids.update(md5_to_gids.get(h, set()))

    if bad_gids:
        samples = [s for s in samples if str(s.gid) not in bad_gids]

    if gid_remap:
        samples = [Sample(s.path, s.cls, gid_remap.get(str(s.gid), str(s.gid)), s.kind) for s in samples]

    groups = {}
    for s in samples:
        groups.setdefault(s.gid, s.cls)
    group_ids = list(groups.keys())
    group_labels = [groups[g] for g in group_ids]
    split = stratified_group_split(group_ids, group_labels, test_size=float(args.test_size), seed=int(args.seed))

    train = [s for s in samples if s.gid in split.train_groups]
    test = [s for s in samples if s.gid in split.test_groups and s.kind == "orig"]

    train_orig_hash: set[str] = set()
    for s in train:
        if s.kind == "orig":
            train_orig_hash.add(_md5_file(s.path))

    kept_test: list[Sample] = []
    removed = 0
    for s in test:
        h = _md5_file(s.path)
        if h in train_orig_hash:
            removed += 1
            continue
        kept_test.append(s)
    test = kept_test

    if not train or not test:
        raise SystemExit("invalid split after dedup (empty train/test)")

    classes = sorted(list({s.cls for s in samples}))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    tfm_train = transforms.Compose(
        [
            transforms.Resize((int(args.img_size), int(args.img_size))),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.25, contrast=0.20, saturation=0.15, hue=0.02),
            transforms.ToTensor(),
        ]
    )
    tfm_test = transforms.Compose(
        [
            transforms.Resize((int(args.img_size), int(args.img_size))),
            transforms.ToTensor(),
        ]
    )

    ds_train = ImgDataset(train, class_to_idx, tfm_train)
    ds_test = ImgDataset(test, class_to_idx, tfm_test)

    use_gpu = torch.cuda.is_available()
    dl_train = DataLoader(ds_train, batch_size=int(args.batch), shuffle=True, num_workers=int(args.num_workers), pin_memory=use_gpu)
    dl_test = DataLoader(ds_test, batch_size=int(args.batch), shuffle=False, num_workers=int(args.num_workers), pin_memory=use_gpu)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, n_classes)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=float(args.lr))

    best_macro = -1.0
    best_path = out_dir / "mobilenetv3_small_best.pt"
    checkpoint_path = out_dir / "checkpoint_latest.pt"
    start_epoch = 1
    epoch_times: list[float] = []

    # Resume from checkpoint if available
    if checkpoint_path.is_file():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_macro = float(ckpt["best_macro"])
        logger.info("deep_resume_checkpoint epoch={} best_macro_f1={:.4f}", start_epoch - 1, best_macro)

    total_epochs = int(args.epochs)

    for ep in range(start_epoch, total_epochs + 1):
        ep_t0 = time.time()
        model.train()
        running_loss = 0.0
        n_batches = 0
        for x, y in tqdm(dl_train, desc=f"train_ep{ep}/{total_epochs}", unit="batch", mininterval=2.0):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            running_loss += float(loss.item())
            n_batches += 1

        avg_loss = running_loss / max(1, n_batches)

        model.eval()
        yt = []
        yp = []
        with torch.no_grad():
            for x, y in tqdm(dl_test, desc=f"eval_ep{ep}/{total_epochs}", unit="batch", mininterval=2.0):
                x = x.to(device)
                out = model(x)
                pred = torch.argmax(out, dim=1).cpu().numpy()
                yp.append(pred)
                yt.append(y.numpy())
        y_true = np.concatenate(yt, axis=0)
        y_pred = np.concatenate(yp, axis=0)
        m = metrics(y_true, y_pred, n_classes=n_classes)

        ep_elapsed = time.time() - ep_t0
        epoch_times.append(ep_elapsed)
        remaining_epochs = total_epochs - ep
        eta_s = float(np.mean(epoch_times)) * remaining_epochs

        is_best = float(m["macro_f1"]) > best_macro
        if is_best:
            best_macro = float(m["macro_f1"])
            torch.save({"model": model.state_dict(), "classes": classes, "img_size": int(args.img_size)}, best_path)

        logger.info(
            "deep_ep={}/{} loss={:.4f} acc={:.4f} macro_f1={:.4f} best={:.4f} ep_s={:.0f} eta_min={:.0f}{}",
            ep, total_epochs, avg_loss, m["accuracy"], m["macro_f1"], best_macro,
            ep_elapsed, eta_s / 60.0,
            " [best]" if is_best else "",
        )

        # Save checkpoint after every epoch (model + optimizer state for safe resume)
        torch.save({
            "epoch": ep,
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "best_macro": best_macro,
            "classes": classes,
            "img_size": int(args.img_size),
        }, checkpoint_path)

    meta = {
        "model": "mobilenet_v3_small",
        "best_macro_f1": best_macro,
        "classes": classes,
        "img_size": int(args.img_size),
        "test_size": float(args.test_size),
        "seed": int(args.seed),
        "dedup_gid_remap_count": int(len(gid_remap)),
        "cross_class_duplicate_groups_removed": int(len(bad_gids)),
        "hash_overlap_removed_from_test": int(removed),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
    }
    (out_dir / "training_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("deep_done best_path={} best_macro_f1={:.4f}", str(best_path), float(best_macro))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
