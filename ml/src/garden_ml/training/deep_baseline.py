from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, precision_recall_fscore_support
from tqdm import tqdm

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
    if torch is None:
        raise SystemExit("install extras: pip install -e '.[dl]'")

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, default="dataset_aug")
    ap.add_argument("--output_dir", type=str, default="artifacts/model_registry/v0001_dl")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.30)
    ap.add_argument("--manifest", type=str, default="augmentation_manifest.csv")
    args = ap.parse_args()

    set_seed(int(args.seed))
    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        rows = samples_from_manifest(dataset_dir, args.manifest, include_kinds={"orig", "aug"}, require_status_ok=True)
        samples = [Sample(Path(p), cls, gid, kind) for p, cls, gid, kind in rows]
    except Exception:
        raw = scan_folder_dataset(dataset_dir)
        samples = [Sample(p, cls, gid, "orig") for p, cls, gid in raw]

    groups = {}
    for s in samples:
        groups.setdefault(s.gid, s.cls)
    group_ids = list(groups.keys())
    group_labels = [groups[g] for g in group_ids]
    split = stratified_group_split(group_ids, group_labels, test_size=float(args.test_size), seed=int(args.seed))

    train = [s for s in samples if s.gid in split.train_groups]
    test = [s for s in samples if s.gid in split.test_groups and s.kind == "orig"]

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

    dl_train = DataLoader(ds_train, batch_size=int(args.batch), shuffle=True, num_workers=2, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=int(args.batch), shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, n_classes)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=float(args.lr))

    best_macro = -1.0
    best_path = out_dir / "mobilenetv3_small_best.pt"

    for ep in range(1, int(args.epochs) + 1):
        model.train()
        for x, y in tqdm(dl_train, desc=f"train_ep{ep}", unit="batch"):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

        model.eval()
        yt = []
        yp = []
        with torch.no_grad():
            for x, y in tqdm(dl_test, desc=f"eval_ep{ep}", unit="batch"):
                x = x.to(device)
                out = model(x)
                pred = torch.argmax(out, dim=1).cpu().numpy()
                yp.append(pred)
                yt.append(y.numpy())
        y_true = np.concatenate(yt, axis=0)
        y_pred = np.concatenate(yp, axis=0)
        m = metrics(y_true, y_pred, n_classes=n_classes)
        logger.info("ep={} acc={:.4f} macro_f1={:.4f}", ep, m["accuracy"], m["macro_f1"])

        if float(m["macro_f1"]) > best_macro:
            best_macro = float(m["macro_f1"])
            torch.save({"model": model.state_dict(), "classes": classes, "img_size": int(args.img_size)}, best_path)

    meta = {"model": "mobilenet_v3_small", "best_macro_f1": best_macro, "classes": classes, "img_size": int(args.img_size)}
    (out_dir / "training_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("saved {}", best_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
