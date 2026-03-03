from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


@dataclass(frozen=True)
class GroupSplit:
    train_groups: set[str]
    test_groups: set[str]


def stratified_group_split(
    group_ids: list[str],
    group_labels: list[str],
    test_size: float,
    seed: int,
) -> GroupSplit:
    g = np.array(group_ids, dtype=object)
    y = np.array(group_labels, dtype=object)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(sss.split(g, y))
    return GroupSplit(train_groups=set(g[train_idx].tolist()), test_groups=set(g[test_idx].tolist()))


def expand_groups(
    samples: Iterable[tuple[str, str, str, str]],
    train_groups: set[str],
    test_groups: set[str],
    test_kinds_only: set[str],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[str], list[str]]:
    train: list[tuple[str, str]] = []
    test: list[tuple[str, str]] = []
    train_groups_exp: list[str] = []
    test_groups_exp: list[str] = []
    for path, cls, gid, kind in samples:
        if gid in train_groups:
            train.append((path, cls))
            train_groups_exp.append(gid)
        elif gid in test_groups and kind in test_kinds_only:
            test.append((path, cls))
            test_groups_exp.append(gid)
    return train, test, train_groups_exp, test_groups_exp
