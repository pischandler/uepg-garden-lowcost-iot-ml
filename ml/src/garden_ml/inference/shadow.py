"""Shadow mode: mirrors /predict traffic to Rust server in background.

How it works
------------
After Python produces its response, `fire_and_log` spawns a daemon thread
that forwards the same image bytes to the Rust /predict endpoint.  The thread
compares both responses and appends one row to a CSV file.

The calling thread (Flask worker) is NOT blocked — the daemon thread runs
independently and is safe to lose on process shutdown.

CSV schema
----------
timestamp_utc, device_id, endpoint,
py_class, py_score, py_total_ms,
rs_class, rs_score, rs_total_ms,
same_class, score_delta, error
"""

from __future__ import annotations

import csv
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

_LOCK = threading.Lock()
_FIELDNAMES = [
    "timestamp_utc", "device_id", "endpoint",
    "py_class", "py_score", "py_total_ms",
    "rs_class", "rs_score", "rs_total_ms",
    "same_class", "score_delta", "error",
]


def _append_row(csv_path: Path, row: dict[str, Any]) -> None:
    with _LOCK:
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
            f.flush()


def _shadow_worker(
    rust_url: str,
    image_bytes: bytes,
    py_class: str | None,
    py_score: float,
    py_total_ms: float,
    device_id: str,
    endpoint: str,
    csv_path: Path,
    timeout_s: int,
) -> None:
    row: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device_id": device_id,
        "endpoint": endpoint,
        "py_class": py_class or "",
        "py_score": round(py_score, 6),
        "py_total_ms": round(py_total_ms, 3),
        "rs_class": "",
        "rs_score": "",
        "rs_total_ms": "",
        "same_class": "",
        "score_delta": "",
        "error": "",
    }
    try:
        t0 = time.perf_counter()
        resp = requests.post(
            f"{rust_url}/predict",
            data=image_bytes,
            headers={"Content-Type": "image/jpeg", "X-Device-Id": device_id},
            timeout=timeout_s,
        )
        rs_total_ms = (time.perf_counter() - t0) * 1000.0
        resp.raise_for_status()
        payload = resp.json()

        rs_class: str = payload.get("classe_predita") or ""
        rs_score: float = float(payload.get("score", 0.0))
        rs_timing: float = float(
            payload.get("timings_ms", {}).get("total_ms", rs_total_ms)
        )

        row["rs_class"] = rs_class
        row["rs_score"] = round(rs_score, 6)
        row["rs_total_ms"] = round(rs_timing, 3)
        row["same_class"] = (py_class or "") == rs_class
        row["score_delta"] = round(abs(py_score - rs_score), 6)
    except Exception as exc:
        row["error"] = str(exc)[:200]

    _append_row(csv_path, row)


def fire_and_log(
    *,
    rust_url: str,
    image_bytes: bytes,
    py_class: str | None,
    py_score: float,
    py_total_ms: float,
    device_id: str,
    endpoint: str,
    csv_path: Path,
    timeout_s: int = 10,
) -> None:
    """Spawn a daemon thread to shadow-call Rust and log the comparison.

    Returns immediately — caller is never blocked.
    """
    t = threading.Thread(
        target=_shadow_worker,
        args=(
            rust_url, image_bytes, py_class, py_score,
            py_total_ms, device_id, endpoint, csv_path, timeout_s,
        ),
        daemon=True,
    )
    t.start()
