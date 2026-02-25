from pathlib import Path
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
ENV = "esp32-s3-devkitc-1"
BUILD_DIR = ROOT / ".pio" / "build" / ENV
SIZES_TXT = BUILD_DIR / "size.txt"
BIN = BUILD_DIR / "firmware.bin"

BUDGET = {
    "firmware_bin_max_bytes": 1_500_000,
}


def parse_size_txt(path: Path) -> dict:
    if not path.exists():
        return {}
    txt = path.read_text(encoding="utf-8", errors="ignore")
    out = {}
    for line in txt.splitlines():
        m = re.search(r"RAM:\s+\[\=+\]\s+([0-9.]+)%\s+\(used\s+(\d+)\s+bytes", line)
        if m:
            out["ram_pct"] = float(m.group(1))
            out["ram_used"] = int(m.group(2))
        m = re.search(r"Flash:\s+\[\=+\]\s+([0-9.]+)%\s+\(used\s+(\d+)\s+bytes", line)
        if m:
            out["flash_pct"] = float(m.group(1))
            out["flash_used"] = int(m.group(2))
    return out


def main() -> int:
    errors = []
    if not BIN.exists():
      print("size budget check skipped: firmware.bin not found")
      return 0

    bin_size = BIN.stat().st_size
    if bin_size > BUDGET["firmware_bin_max_bytes"]:
        errors.append(
            f"firmware.bin too large: {bin_size} > {BUDGET['firmware_bin_max_bytes']}"
        )

    sizes = parse_size_txt(SIZES_TXT)
    report = {
        "bin_size": bin_size,
        "budget": BUDGET,
        "size_report": sizes,
    }
    (ROOT / "artifacts_size.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if errors:
        for e in errors:
            print(e)
        return 1
    print("size budget check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
