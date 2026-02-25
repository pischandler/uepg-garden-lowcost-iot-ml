from __future__ import annotations

import os
import time
import requests


BASE = os.getenv("HIL_BASE_URL", "http://192.168.4.1")
TIMEOUT = float(os.getenv("HIL_TIMEOUT", "6"))


def get_json(path: str) -> dict:
    r = requests.get(BASE + path, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def post_json(path: str, payload: dict) -> dict:
    r = requests.post(BASE + path, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def main() -> int:
    checks = [
        ("/health", lambda d: "online" in d),
        ("/status", lambda d: "framesize" in d),
        ("/api/sensors", lambda d: "soil_pct" in d),
        ("/api/irrigation", lambda d: "pump_on" in d),
        ("/api/inference/status", lambda d: "queued" in d),
    ]
    for path, pred in checks:
        data = get_json(path)
        if not pred(data):
            print(f"failed check: {path}")
            return 1

    post_json("/api/inference/run", {})
    time.sleep(1.0)
    _ = get_json("/api/inference/last")

    print("HIL smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
