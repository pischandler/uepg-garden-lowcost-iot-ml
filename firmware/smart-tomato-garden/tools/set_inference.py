#!/usr/bin/env python3
"""
Configure the inference server (ML API) on the ESP from the host.
Usage:
  python tools/set_inference.py --esp 192.168.100.12 --ml 192.168.100.11
  python tools/set_inference.py --esp 192.168.100.12 --ml 192.168.100.11 --port 5000 --path /predict
"""
from __future__ import annotations

import argparse
import os
import sys

try:
    import requests
except ImportError:
    print("Install requests: pip install requests", file=sys.stderr)
    sys.exit(1)

TIMEOUT = float(os.getenv("STG_TIMEOUT", "10"))


def main() -> int:
    p = argparse.ArgumentParser(description="Set inference server (host/port/path) on the ESP via API.")
    p.add_argument("--esp", required=True, help="ESP device IP (e.g. 192.168.100.12)")
    p.add_argument("--ml", required=True, help="ML API host IP or hostname (e.g. 192.168.100.11)")
    p.add_argument("--port", type=int, default=5000, help="ML API port (default: 5000)")
    p.add_argument("--path", default="/predict", help="ML API path (default: /predict)")
    args = p.parse_args()

    base = f"http://{args.esp}"
    url = f"{base}/api/inference/config"
    body = {"infer_host": args.ml, "infer_port": args.port, "infer_path": args.path}

    try:
        r = requests.post(url, json=body, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        print("OK: inference config updated on ESP")
        print(f"  infer_host={data.get('infer_host', '')}")
        print(f"  infer_port={data.get('infer_port', '')}")
        print(f"  infer_path={data.get('infer_path', '')}")
        return 0
    except requests.RequestException as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
