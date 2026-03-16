"""Shadow mode report: summarise agreement between Python and Rust predictions.

Run from repo root:
    python ml/scripts/shadow_report.py
    python ml/scripts/shadow_report.py --csv salvas/shadow_log.csv --out ml/artifacts/shadow_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="salvas/shadow_log.csv", help="Shadow log CSV path")
    ap.add_argument("--out", default=None, help="Write JSON report here (optional)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: shadow log not found: {csv_path}", file=sys.stderr)
        print("Enable shadow mode with GML_SHADOW_MODE=1 and GML_SHADOW_URL=http://rust:5002", file=sys.stderr)
        return 1

    rows = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("Shadow log is empty — no requests captured yet.")
        return 0

    n_total = len(rows)
    n_error = sum(1 for r in rows if r.get("error"))
    n_ok = n_total - n_error

    same = [r for r in rows if r.get("same_class") == "True"]
    diff = [r for r in rows if r.get("same_class") == "False"]

    agreement_rate = len(same) / n_ok if n_ok else 0.0

    score_deltas = []
    for r in rows:
        try:
            score_deltas.append(float(r["score_delta"]))
        except (ValueError, KeyError):
            pass

    py_latencies = []
    rs_latencies = []
    for r in rows:
        try:
            py_latencies.append(float(r["py_total_ms"]))
        except (ValueError, KeyError):
            pass
        try:
            rs_latencies.append(float(r["rs_total_ms"]))
        except (ValueError, KeyError):
            pass

    def _avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    print(f"\n{'='*60}")
    print(f"Shadow Mode Report — {csv_path}")
    print(f"{'='*60}")
    print(f"  Total requests :  {n_total}")
    print(f"  Rust errors    :  {n_error}  ({n_error/n_total*100:.1f}%)")
    print(f"  Compared       :  {n_ok}")
    print(f"  Agreement      :  {len(same)}/{n_ok}  ({agreement_rate*100:.1f}%)")
    print(f"  Divergences    :  {len(diff)}")

    if score_deltas:
        print(f"\n  Score delta    :  avg={_avg(score_deltas):.4f}  max={max(score_deltas):.4f}")

    if py_latencies and rs_latencies:
        avg_py = _avg(py_latencies)
        avg_rs = _avg(rs_latencies)
        print(f"\n  Avg latency    :  Python={avg_py:.1f}ms  Rust={avg_rs:.1f}ms")
        if avg_py > 0:
            print(f"  Speedup        :  {avg_py/avg_rs:.1f}×" if avg_rs > 0 else "  Speedup        :  N/A")

    if diff:
        print(f"\n  --- Divergent predictions (first 10) ---")
        for r in diff[:10]:
            print(f"    [{r.get('timestamp_utc','')}] "
                  f"py={r.get('py_class','')} ({float(r.get('py_score',0)):.3f})  "
                  f"rs={r.get('rs_class','')} ({float(r.get('rs_score',0)):.3f})  "
                  f"device={r.get('device_id','')}")

    # Cutover recommendation
    print(f"\n{'='*60}")
    if n_ok >= 100 and agreement_rate >= 0.98 and n_error / n_total <= 0.01:
        print("✓ CUTOVER RECOMMENDED")
        print("  Agreement ≥98%, error rate ≤1%, n≥100 requests.")
        print("  Run:  make cutover-rust DEVICE_IP=<esp32-ip>")
    elif n_ok < 100:
        print(f"  Collecting data: {n_ok}/100 requests compared so far.")
    else:
        print("✗ NOT READY FOR CUTOVER")
        if agreement_rate < 0.98:
            print(f"  Agreement {agreement_rate*100:.1f}% < 98% — investigate divergences above.")
        if n_error / n_total > 0.01:
            print(f"  Error rate {n_error/n_total*100:.1f}% > 1% — Rust server unstable.")
    print(f"{'='*60}\n")

    if args.out:
        report = {
            "n_total": n_total,
            "n_error": n_error,
            "n_ok": n_ok,
            "agreement_rate": round(agreement_rate, 4),
            "n_same_class": len(same),
            "n_divergent": len(diff),
            "avg_score_delta": round(_avg(score_deltas), 6),
            "avg_py_latency_ms": round(_avg(py_latencies), 3),
            "avg_rs_latency_ms": round(_avg(rs_latencies), 3),
            "cutover_ready": (n_ok >= 100 and agreement_rate >= 0.98 and n_error / n_total <= 0.01),
            "divergences": diff[:50],
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"Report written to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
