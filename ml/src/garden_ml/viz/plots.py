"""Plot evaluation outputs (calibration curve, confusion matrix). Requires [viz]: pip install -e '.[viz]'."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_eval_plots(out_dir: Path, eval_result: dict[str, Any]) -> None:
    """Write calibration_curve.png and confusion_matrix.png to out_dir when data is present."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "calibration_bins" in eval_result and eval_result["calibration_bins"]:
        _plot_calibration_curve(out_dir, eval_result["calibration_bins"])

    if "confusion_matrix" in eval_result and "classes" in eval_result:
        cm = np.asarray(eval_result["confusion_matrix"])
        classes = list(eval_result["classes"])
        if cm.size > 0 and classes:
            _plot_confusion_matrix(out_dir, cm, classes)


def _plot_calibration_curve(out_dir: Path, bins: list[dict[str, float]]) -> None:
    centers = [b["bin_center"] for b in bins]
    accs = [b["accuracy"] for b in bins]
    confs = [b["confidence"] for b in bins]
    counts = [b["count"] for b in bins]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", lw=1)
    ax.plot(confs, accs, "s-", label="Model", color="C0", markeredgecolor="white", markersize=6)
    ax.set_xlabel("Mean predicted probability (confidence)")
    ax.set_ylabel("Fraction of positives (accuracy)")
    ax.set_title("Calibration curve")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    if sum(counts) > 0:
        ax.annotate(f"n={int(sum(counts))} samples, {len(bins)} bins", xy=(0.02, 0.98), xycoords="axes fraction", fontsize=9, va="top")
    fig.tight_layout()
    fig.savefig(out_dir / "calibration_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion_matrix(out_dir: Path, cm: np.ndarray, classes: list[str]) -> None:
    n_classes = len(classes)
    if cm.shape != (n_classes, n_classes):
        return
    fig, ax = plt.subplots(figsize=(max(6, n_classes * 0.6), max(5, n_classes * 0.5)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(classes, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(classes)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
