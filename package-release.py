"""
Package the Gemma4-LLM release zip.

Creates: dist/Gemma4-LLM.zip containing:
  Gemma4-LLM/
  ├── Gemma4-LLM.exe
  └── models/
      ├── gemma-4-E2B-it-Q4_K_M.gguf
      └── mmproj-F16.gguf
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
DIST = ROOT / "dist"
EXE = DIST / "Gemma4-LLM.exe"
MODELS_DIR = ROOT / "models"
OUT_ZIP = DIST / "Gemma4-LLM.zip"

FILES = [
    (EXE, "Gemma4-LLM/Gemma4-LLM.exe"),
    (MODELS_DIR / "gemma-4-E2B-it-Q4_K_M.gguf", "Gemma4-LLM/models/gemma-4-E2B-it-Q4_K_M.gguf"),
    (MODELS_DIR / "mmproj-F16.gguf", "Gemma4-LLM/models/mmproj-F16.gguf"),
]


def main():
    for src, _ in FILES:
        if not src.exists():
            print(f"[ERROR] Missing: {src}")
            return 1

    print("Packaging Gemma4-LLM release zip...")
    print()

    DIST.mkdir(exist_ok=True)
    total = 0

    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_STORED) as zf:
        for src, arc_name in FILES:
            size_mb = src.stat().st_size / 1024 / 1024
            total += src.stat().st_size
            print(f"  adding  {arc_name:<50} ({size_mb:>7.1f} MB)")
            zf.write(src, arc_name)

    zip_mb = OUT_ZIP.stat().st_size / 1024 / 1024
    print()
    print(f"  Output: {OUT_ZIP}")
    print(f"  Size:   {zip_mb:,.1f} MB ({total / 1024 / 1024:,.1f} MB uncompressed)")
    return 0


if __name__ == "__main__":
    exit(main())
