from pathlib import Path
import io
import gzip
import hashlib

def gzip_deterministic(data: bytes) -> bytes:
  buf = io.BytesIO()
  with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9, mtime=0) as f:
    f.write(data)
  return buf.getvalue()

def to_c_array(data: bytes, name: str, per_line: int = 16) -> str:
  rows = []
  for i in range(0, len(data), per_line):
    chunk = data[i:i + per_line]
    rows.append(", ".join(f"0x{b:02x}" for b in chunk))
  body = ",\n  ".join(rows)
  return f"static const uint8_t {name}[] PROGMEM = {{\n  {body}\n}};\n"

def build(project_dir: Path) -> None:
  web_dir = project_dir / "web"
  inc_dir = project_dir / "include"

  html_path = web_dir / "index.html"
  gz_path = web_dir / "index.html.gz"
  out_h = inc_dir / "camera_index.h"

  html = html_path.read_bytes()
  gz = gzip_deterministic(html)
  gz_path.write_bytes(gz)

  sha1 = hashlib.sha1(gz).hexdigest()

  header = (
    "#pragma once\n\n"
    "#include <Arduino.h>\n"
    "#include <pgmspace.h>\n\n"
    f"static const char INDEX_HTML_SHA1[] PROGMEM = \"{sha1}\";\n"
    f"static const size_t INDEX_HTML_GZ_LEN = {len(gz)};\n\n"
    + to_c_array(gz, "INDEX_HTML_GZ")
  )

  out_h.write_text(header, encoding="utf-8")

def main() -> None:
  project_dir = Path(__file__).resolve().parents[1]
  build(project_dir)

try:
  Import("env")
  def _pio_action(source, target, env):
    build(Path(env["PROJECT_DIR"]))
  env.AddPreAction("buildprog", _pio_action)
except Exception:
  if __name__ == "__main__":
    main()
