"""Plot a visual comparison of Flask (Python) vs Axum (Rust) server metrics.

Run from the repo root:
    python ml/scripts/plot_migration_comparison.py
    python ml/scripts/plot_migration_comparison.py --out docs/migration_benchmark.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Benchmark data (measured) ─────────────────────────────────────────────────
# Python/Flask: measured on same machine, gunicorn single-worker
# Rust/Axum:   measured with release binary (cargo build --release)

DATA = {
    # ── Latency (ms per request, single image 128×128) ──────────────────────
    "latency_features_ms": {
        "Python (Flask)": 30,      # ~30ms feature extraction in Python
        "Rust (Axum)":    88,      # release build (--release)
    },
    "latency_predict_ms": {
        "Python (Flask)": 15,      # sklearn XGBoost inference
        "Rust (Axum)":    0.5,     # ONNX Runtime ~0.5ms
    },
    "latency_total_ms": {
        "Python (Flask)": 46,      # ~30ms feat + 15ms sklearn + HTTP
        "Rust (Axum)":    88,
    },

    # ── Memory (MB RSS at idle, single-worker) ───────────────────────────────
    "memory_mb": {
        "Python (Flask)": 210,     # gunicorn + sklearn + numpy + mahotas + skimage
        "Rust (Axum)":    38,      # stripped release binary
    },

    # ── Binary / image size (MB) ──────────────────────────────────────────────
    "binary_size_mb": {
        "Python (Flask)": 0,       # runtime — not a binary
        "Rust (Axum)":    27,
    },

    # ── Startup time (seconds) ────────────────────────────────────────────────
    "startup_s": {
        "Python (Flask)": 4.5,     # gunicorn + Python imports
        "Rust (Axum)":    0.23,
    },
}

COLORS = {
    "Python (Flask)": "#4C72B0",
    "Rust (Axum)":    "#55A868",
}

LABELS = {
    "latency_features_ms": ("Feature extraction latency (ms)", "Lower is better", "ms"),
    "latency_total_ms":    ("Total request latency (ms)", "Lower is better", "ms"),
    "memory_mb":           ("Memory at idle (MB RSS)", "Lower is better", "MB"),
    "binary_size_mb":      ("Binary / image size (MB)", "Lower is better", "MB"),
    "startup_s":           ("Server startup time (s)", "Lower is better", "s"),
}


def _bar_group(ax: plt.Axes, metric: str, title: str, subtitle: str, unit: str) -> None:
    entries = DATA[metric]
    names = list(entries.keys())
    values = list(entries.values())
    colors = [COLORS[n] for n in names]

    x = np.arange(len(names))
    bars = ax.bar(x, values, color=colors, width=0.55, edgecolor="white", linewidth=0.8)

    # Value labels on top of each bar
    for bar, val in zip(bars, values):
        if val == 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                "runtime",
                ha="center", va="bottom", fontsize=8, color="#888",
            )
        else:
            label = f"{val:.0f}{unit}" if val >= 1 else f"{val:.1f}{unit}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                label,
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_title(f"{title}\n({subtitle})", fontsize=10, pad=8)
    ax.set_ylabel(unit, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(v for v in values if v > 0) * 1.25)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ml/artifacts/migration_benchmark.png",
                    help="Output PNG path")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    fig.suptitle(
        "Flask/Python  →  Axum/Rust — Migration Benchmark\n"
        "Smart Tomato Garden Inference Server",
        fontsize=13, fontweight="bold", y=1.02,
    )

    metrics = [
        ("latency_features_ms", *LABELS["latency_features_ms"]),
        ("latency_total_ms",    *LABELS["latency_total_ms"]),
        ("memory_mb",           *LABELS["memory_mb"]),
        ("binary_size_mb",      *LABELS["binary_size_mb"]),
        ("startup_s",           *LABELS["startup_s"]),
    ]

    for ax, (metric, title, subtitle, unit) in zip(axes, metrics):
        _bar_group(ax, metric, title, subtitle, unit)

    # Legend
    handles = [mpatches.Patch(color=COLORS[k], label=k) for k in COLORS]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.08), fontsize=10, frameon=False)

    # Improvement annotations (Python → Rust release)
    improvements = {
        "latency_features_ms": (30, 88, "note: Rust release\nis still 3× faster total"),
        "latency_total_ms":    (46, 88, "~2× faster\n(ONNX vs sklearn)"),
        "memory_mb":           (210, 38, "~5.5× less RAM"),
        "startup_s":           (4.5, 0.23, "~20× faster start"),
    }

    # Add summary box
    summary_lines = [
        "Key wins (Python → Rust release):",
        f"  • Memory:  210 MB → 38 MB  (5.5×↓)",
        f"  • Startup: 4.5 s  → 0.23 s (20×↓)",
        f"  • Binary:  runtime → 27 MB  (self-contained)",
        f"  • ONNX inference:  ~0.5 ms  (vs 15 ms sklearn)",
        "",
        "Note: Rust feature extraction (88 ms) is",
        "slower than Python (30 ms) — Zernike + Haralick",
        "reimplementation overhead vs C extensions.",
    ]
    fig.text(
        0.5, -0.22,
        "\n".join(summary_lines),
        ha="center", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8", edgecolor="#ccc"),
    )

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
