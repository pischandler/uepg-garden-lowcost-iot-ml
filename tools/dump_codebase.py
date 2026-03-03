from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_INCLUDE = {
    ".py",
    ".pyi",
    ".txt",
    ".md",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".ini",
    ".cfg",
    ".env",
    ".dockerfile",
    ".ps1",
    ".sh",
    ".bat",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".vue",
    ".svelte",
    ".html",
    ".css",
    ".scss",
    ".sql",
    ".csv",
}

DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".nuxt",
    ".idea",
    ".vscode",
    "datasets",
    "dataset",
    "artifacts",
    "mlruns",
    "reports",
}

SEPARATOR = "\n\n" + ("=" * 90) + "\n\n"


def is_included_file(path: Path, include_exts: set[str]) -> bool:
    name = path.name.lower()
    if name == "dockerfile":
        return True
    if name.startswith(".") and "." not in name[1:]:
        # dotfiles like ".env"
        return True
    return path.suffix.lower() in include_exts


def read_text_safe(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except Exception:
            return None
    except Exception:
        return None


def collect_files(root: Path, exclude_dirs: set[str], include_exts: set[str], max_bytes: int) -> list[Path]:
    files: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        base = Path(dirpath)

        for filename in filenames:
            path = base / filename
            if not is_included_file(path, include_exts):
                continue
            try:
                if path.stat().st_size > max_bytes:
                    continue
            except Exception:
                continue
            files.append(path)

    return sorted(files, key=lambda p: str(p).lower())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dump de codigo: caminhos + conteudo em um unico TXT."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Pasta raiz do projeto (default: .)",
    )
    parser.add_argument(
        "-o",
        "--out",
        default="dump.txt",
        help="Arquivo de saida (default: dump.txt)",
    )
    parser.add_argument(
        "--max-kb",
        type=int,
        default=512,
        help="Tamanho maximo por arquivo em KB (default: 512)",
    )
    parser.add_argument(
        "--no-default-excludes",
        action="store_true",
        help="Nao excluir pastas padrao",
    )
    parser.add_argument(
        "--extra-exclude",
        action="append",
        default=[],
        help="Adicionar pasta para excluir (pode repetir)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_path = Path(args.out).resolve()
    max_bytes = int(args.max_kb) * 1024

    exclude_dirs = set() if args.no_default_excludes else set(DEFAULT_EXCLUDE_DIRS)
    exclude_dirs.update(set(args.extra_exclude))

    files = collect_files(root=root, exclude_dirs=exclude_dirs, include_exts=DEFAULT_INCLUDE, max_bytes=max_bytes)

    parts: list[str] = [f"ROOT: {root}\nFILES: {len(files)}\n"]
    written_files = 0

    for file_path in files:
        if file_path == out_path:
            continue
        rel = file_path.relative_to(root)
        text = read_text_safe(file_path)
        if text is None:
            continue
        parts.append(f"PATH: {rel}\n")
        parts.append(text.rstrip() + "\n")
        parts.append(SEPARATOR)
        written_files += 1

    out_path.write_text("".join(parts), encoding="utf-8")
    print(
        f"OK: wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB), "
        f"files included: {written_files}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
