from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
CAMERA_INDEX = ROOT / "include" / "camera_index.h"
OUT = ROOT / "artifacts_meta.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(65536)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> None:
    data = {
        "camera_index_sha256": sha256_file(CAMERA_INDEX) if CAMERA_INDEX.exists() else "",
    }
    OUT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("artifact metadata exported")


if __name__ == "__main__":
    main()
