from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from garden_ml.config.constants import DEFAULT_AUG_MANIFEST
from garden_ml.data.io import is_image_file

_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def normalize_group_base(stem: str) -> str:
    base = stem.split("__")[0] if "__" in stem else stem
    base = base.strip()
    if "___" in base:
        left, right = base.split("___", 1)
        if _UUID_RE.match(left.strip()):
            base = right.strip()
    base = re.sub(r"\s+", " ", base)
    return base


def make_group_id(cls: str, stem: str) -> str:
    return f"{cls}/{normalize_group_base(stem)}"


def normalize_group_id(cls: str, gid: str) -> str:
    g = (gid or "").strip()
    if not g:
        return make_group_id(cls, "")
    if "/" in g:
        left, right = g.split("/", 1)
        base = right.strip() if left.strip() else g.strip()
    else:
        base = g
    return make_group_id(cls, base)


@dataclass(frozen=True)
class AugRow:
    cls: str
    group_id: str
    source_path: str
    output_path: str
    kind: str
    aug_index: str
    seed: str
    status: str
    error: str


def load_augmentation_manifest(dataset_dir: Path, manifest_name: str = DEFAULT_AUG_MANIFEST) -> list[AugRow]:
    manifest_path = dataset_dir / manifest_name
    if not manifest_path.is_file():
        raise FileNotFoundError(str(manifest_path))

    out: list[AugRow] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        need = {"class", "group_id", "source_path", "output_path", "kind", "aug_index", "seed", "status", "error"}
        if not need.issubset(set(r.fieldnames or [])):
            raise ValueError(f"manifest invalid columns: {r.fieldnames}")
        for row in r:
            out.append(
                AugRow(
                    cls=(row.get("class") or "").strip(),
                    group_id=(row.get("group_id") or "").strip(),
                    source_path=(row.get("source_path") or "").strip(),
                    output_path=(row.get("output_path") or "").strip(),
                    kind=(row.get("kind") or "").strip().lower(),
                    aug_index=(row.get("aug_index") or "").strip(),
                    seed=(row.get("seed") or "").strip(),
                    status=(row.get("status") or "").strip().lower(),
                    error=(row.get("error") or "").strip(),
                )
            )
    return out


def scan_folder_dataset(dataset_dir: Path) -> list[tuple[Path, str, str]]:
    rows: list[tuple[Path, str, str]] = []
    classes = sorted([p.name for p in dataset_dir.iterdir() if p.is_dir()])
    for c in classes:
        class_dir = dataset_dir / c
        for p in sorted([x for x in class_dir.iterdir() if x.is_file() and is_image_file(x)]):
            group_id = make_group_id(c, p.stem)
            rows.append((p, c, group_id))
    return rows


def samples_from_manifest(
    dataset_dir: Path,
    manifest_name: str,
    include_kinds: set[str],
    require_status_ok: bool = True,
) -> list[tuple[Path, str, str, str]]:
    rows = load_augmentation_manifest(dataset_dir, manifest_name)
    out: list[tuple[Path, str, str, str]] = []
    for r in rows:
        if require_status_ok and r.status != "ok":
            continue
        if r.kind not in include_kinds:
            continue
        if not r.cls or not r.output_path:
            continue
        p = dataset_dir / r.output_path
        if p.is_file() and is_image_file(p):
            gid = normalize_group_id(r.cls, r.group_id) if r.group_id else make_group_id(r.cls, Path(r.output_path).stem)
            out.append((p, r.cls, gid, r.kind))
    return out


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def class_distribution(items: Iterable[tuple[str, str]]) -> dict[str, int]:
    d: dict[str, int] = {}
    for cls, _ in items:
        d[cls] = d.get(cls, 0) + 1
    return dict(sorted(d.items(), key=lambda x: x[0]))
