"""Gera figura comparando XGBoost vs SVM vs Random Forest.

Usa model_comparison.csv do model registry v0005.

Uso:
    python ml/scripts/plot_classifier_comparison.py
    python ml/scripts/plot_classifier_comparison.py --version v0005 --out ml/artifacts/classifier_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = ["#4C72B0", "#DD8452", "#55A868"]

METRICS = [
    ("test_macro_f1",          "Macro F1"),
    ("test_mcc",               "MCC"),
    ("test_balanced_accuracy", "Balanced Accuracy"),
]


def plot(csv_path: Path, out: Path) -> None:
    df = pd.read_csv(csv_path)

    # Sort: XGBoost first, then SVM, then RF (by macro F1 desc)
    df = df.sort_values("test_macro_f1", ascending=False).reset_index(drop=True)

    models = df["model"].tolist()
    x = np.arange(len(models))
    n_metrics = len(METRICS)
    width = 0.22
    offsets = np.linspace(-(n_metrics - 1) * width / 2, (n_metrics - 1) * width / 2, n_metrics)

    fig, ax = plt.subplots(figsize=(10, 6))

    for (col, label), offset, color in zip(METRICS, offsets, COLORS):
        vals = df[col].tolist()
        bars = ax.bar(x + offset, vals, width=width, color=color, alpha=0.88,
                      edgecolor="white", linewidth=0.7, label=label)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylim(0.85, 1.02)
    ax.set_title(
        "Comparação de Classificadores — Smart Tomato Garden\n"
        "Mesmo conjunto de 188 features e split StratifiedGroupKFold",
        fontsize=15, fontweight="bold", pad=12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.01))
    ax.grid(axis="y", linestyle="--", alpha=0.4, which="both")
    ax.legend(fontsize=13, frameon=False)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Salvo: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v0005")
    ap.add_argument("--out", default="ml/artifacts/classifier_comparison.png")
    args = ap.parse_args()

    csv_path = Path(f"ml/artifacts/model_registry/{args.version}/model_comparison.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Não encontrado: {csv_path}")

    plot(csv_path, Path(args.out))


if __name__ == "__main__":
    main()
