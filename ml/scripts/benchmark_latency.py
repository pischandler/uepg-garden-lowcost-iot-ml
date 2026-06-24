"""Benchmark de latência: Python/Flask vs Rust/Axum.

Coleta N medições por servidor e gera 3 painéis para artigo científico:
  1. Violin plot — wall-clock latency (o que o ESP32 experimenta)
  2. Stacked bar — breakdown features_ms / predict_ms / overhead
  3. ECDF — distribuição acumulada de latência

Uso:
    # Dentro do Docker (URLs via container name):
    python ml/scripts/benchmark_latency.py \
        --python-url http://garden-ml-api:5000/predict \
        --rust-url   http://garden-ml-rust:5002/predict

    # No host (URLs via localhost):
    python ml/scripts/benchmark_latency.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import requests
from scipy import stats

DEFAULT_PYTHON_URL = "http://localhost:5000/predict"
DEFAULT_RUST_URL   = "http://localhost:5002/predict"

COLORS = {
    "Python (Flask)": "#4C72B0",
    "Rust (Axum)":    "#55A868",
}
BREAKDOWN_COLORS = {
    "features_ms": "#6A9CC9",
    "predict_ms":  "#E07B54",
    "overhead_ms": "#B0B0B0",
}

DATASET_DIR = Path("tools/dataset/plantvillage_tomato/raw")


def _collect_images(n: int) -> list[bytes]:
    raw: list[bytes] = []
    for cls_dir in sorted(DATASET_DIR.iterdir()):
        if cls_dir.is_dir():
            for jpg in sorted(cls_dir.glob("*.jpg"))[:5]:
                raw.append(jpg.read_bytes())
    if not raw:
        raise RuntimeError(f"Nenhuma imagem encontrada em {DATASET_DIR}")
    # cycle through available images to fill n slots
    images = [raw[i % len(raw)] for i in range(n)]
    print(f"  ({len(raw)} imagens únicas, cicladas para {n} requisições)")
    return images


def _request(url: str, img: bytes, session: requests.Session) -> tuple[float, float, float]:
    """Return (wall_ms, features_ms, predict_ms)."""
    t0 = time.perf_counter()
    resp = session.post(url, data=img, headers={"Content-Type": "image/jpeg"}, timeout=30)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    resp.raise_for_status()
    timings = resp.json().get("timings_ms", {})
    return wall_ms, float(timings.get("features_ms", 0.0)), float(timings.get("predict_ms", 0.0))


def benchmark(
    n: int,
    warmup: int,
    python_url: str,
    rust_url: str,
) -> dict[str, dict[str, list[float]]]:
    warmup_images = _collect_images(warmup)
    bench_images  = _collect_images(n)
    print(f"warmup={warmup}  |  N={n}")

    servers = {"Python (Flask)": python_url, "Rust (Axum)": rust_url}
    results: dict[str, dict[str, list[float]]] = {}

    for name, url in servers.items():
        print(f"\n→ {name}")
        wall_all, feat_all, pred_all = [], [], []
        with requests.Session() as sess:
            print(f"  warmup...", end=" ", flush=True)
            for img in warmup_images:
                try:
                    _request(url, img, sess)
                except Exception as e:
                    print(f"\n  AVISO warmup: {e}")
            print("ok")

            for i, img in enumerate(bench_images):
                try:
                    w, f, p = _request(url, img, sess)
                    wall_all.append(w)
                    feat_all.append(f)
                    pred_all.append(p)
                    if (i + 1) % 25 == 0:
                        print(f"  {i+1}/{n}  wall médio={np.mean(wall_all):.1f}ms")
                except Exception as e:
                    print(f"  ERRO req {i+1}: {e}")

        results[name] = {
            "wall_ms":     wall_all,
            "features_ms": feat_all,
            "predict_ms":  pred_all,
        }
        mu, sigma = np.mean(wall_all), np.std(wall_all)
        p50, p95 = np.percentile(wall_all, 50), np.percentile(wall_all, 95)
        print(f"  wall: {mu:.1f} ± {sigma:.1f} ms  (p50={p50:.1f}, p95={p95:.1f})")

    return results


def _mann_whitney_annotation(ax: plt.Axes, a: list[float], b: list[float]) -> None:
    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    label = f"p {'< 0,001' if p < 0.001 else f'= {p:.3f}'} (Mann-Whitney U)"
    ax.text(0.5, 0.03, label, transform=ax.transAxes,
            fontsize=13, ha="center", color="#444",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="#ccc", alpha=0.9))


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: {path}")


def plot(results: dict[str, dict[str, list[float]]], out_base: Path, n: int) -> None:
    names = list(results.keys())

    # ── Figura 1: Violin plot (wall-clock) ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    wall_data = [results[nm]["wall_ms"] for nm in names]
    parts = ax.violinplot(wall_data, positions=range(len(names)),
                          showmedians=True, showextrema=True, widths=0.6)
    for body, nm in zip(parts["bodies"], names):
        body.set_facecolor(COLORS[nm])
        body.set_alpha(0.78)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2.5)
    for line in ["cbars", "cmins", "cmaxes"]:
        parts[line].set_color("#777")
        parts[line].set_linewidth(1.5)

    top = max(max(v) for v in wall_data)
    for i, vals in enumerate(wall_data):
        med = np.median(vals)
        std = np.std(vals)
        ax.text(i, top * 1.04,
                f"mediana {med:.1f} ms\n±{std:.1f} ms dp",
                ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=14)
    ax.set_ylabel("Latência wall-clock (ms)", fontsize=14)
    ax.set_title(
        f"Distribuição da latência — Flask/Python vs. Axum/Rust\n(N={n} requisições por servidor, imagens 128×128 px)",
        fontsize=15, fontweight="bold", pad=12,
    )
    ax.set_ylim(0, top * 1.22)
    ax.tick_params(axis="y", labelsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _mann_whitney_annotation(ax, wall_data[0], wall_data[1])
    fig.tight_layout()
    _save(fig, out_base.parent / (out_base.stem + "_violin.png"))

    # ── Figura 2: Stacked bar — breakdown ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    feat_means = [np.mean(results[nm]["features_ms"]) for nm in names]
    pred_means = [np.mean(results[nm]["predict_ms"])  for nm in names]
    wall_means = [np.mean(results[nm]["wall_ms"])     for nm in names]
    overhead   = [max(0.0, w - f - p) for w, f, p in zip(wall_means, feat_means, pred_means)]

    x = np.arange(len(names))
    w = 0.50
    ax.bar(x, feat_means, w, label="Extração de features", color=BREAKDOWN_COLORS["features_ms"], alpha=0.92)
    ax.bar(x, pred_means, w, bottom=feat_means,
           label="Inferência (ONNX / sklearn)", color=BREAKDOWN_COLORS["predict_ms"], alpha=0.92)
    ax.bar(x, overhead, w, bottom=[f + p for f, p in zip(feat_means, pred_means)],
           label="Overhead (rede + I/O)", color=BREAKDOWN_COLORS["overhead_ms"], alpha=0.80)

    for i, (f, p, o) in enumerate(zip(feat_means, pred_means, overhead)):
        total = f + p + o
        ax.text(i, total + 0.4, f"{total:.1f} ms", ha="center", va="bottom",
                fontsize=14, fontweight="bold")
        if f > 1.5:
            ax.text(i, f / 2, f"{f:.1f} ms", ha="center", va="center",
                    fontsize=12, color="white", fontweight="bold")
        if p > 0.4:
            ax.text(i, f + p / 2, f"{p:.1f} ms", ha="center", va="center",
                    fontsize=12, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylabel("Latência média (ms)", fontsize=14)
    ax.set_title(
        "Decomposição da latência por etapa\n(features + inferência + overhead)",
        fontsize=15, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=13, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, out_base.parent / (out_base.stem + "_breakdown.png"))

    # ── Figura 3: ECDF ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for nm in names:
        vals = np.sort(results[nm]["wall_ms"])
        ecdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, ecdf, color=COLORS[nm], linewidth=2.5, label=nm)
        p95 = np.percentile(vals, 95)
        ax.axvline(p95, color=COLORS[nm], linestyle="--", linewidth=1.5, alpha=0.7)
        offset = 0.06 if nm == names[0] else 0.14
        ax.text(p95 + 0.4, offset, f"p95 = {p95:.0f} ms",
                fontsize=12, color=COLORS[nm], fontweight="bold")

    ax.axhline(0.95, color="#bbb", linestyle=":", linewidth=1.2)
    ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] > 0 else 1,
            0.965, "95° percentil", fontsize=12, color="#888")
    ax.set_xlabel("Latência wall-clock (ms)", fontsize=14)
    ax.set_ylabel("P(latência ≤ t)", fontsize=14)
    ax.set_title(
        "Distribuição acumulada empírica (ECDF) — latência wall-clock",
        fontsize=15, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=13, frameon=False)
    ax.tick_params(axis="both", labelsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    _save(fig, out_base.parent / (out_base.stem + "_ecdf.png"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",          type=int, default=100)
    ap.add_argument("--warmup",     type=int, default=10)
    ap.add_argument("--python-url", default=DEFAULT_PYTHON_URL)
    ap.add_argument("--rust-url",   default=DEFAULT_RUST_URL)
    ap.add_argument("--out",        default="ml/artifacts/latency_benchmark.png")
    ap.add_argument("--json-out",   default="ml/artifacts/latency_benchmark_raw.json")
    args = ap.parse_args()

    # Se já existe raw data salvo, reutiliza sem bater nos servidores
    json_out = Path(args.json_out)
    if json_out.exists() and args.n == 100:
        print(f"Carregando raw data existente: {json_out}")
        results = json.loads(json_out.read_text())
    else:
        results = benchmark(args.n, args.warmup, args.python_url, args.rust_url)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(results, indent=2))
        print(f"Raw data: {json_out}")

    plot(results, Path(args.out), args.n)


if __name__ == "__main__":
    main()
