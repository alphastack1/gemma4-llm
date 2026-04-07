"""
Stage the minimal bin_exe/ folder for EXE bundling.

Copies only the files llama-server needs at runtime:
  - llama-server.exe + required engine DLLs
  - ggml-cuda.dll + ggml-rpc.dll (stock llama.cpp binaries)
  - All ggml-cpu-*.dll arch variants (auto-selected at runtime)
  - mtmd.dll (multimodal projection support)
  - CUDA runtime DLLs (cublas*, cudart*) needed by ggml-cuda.dll
  - libomp140.x86_64.dll (OpenMP threading)

The CUDA DLLs load fine on any Windows machine even without an
NVIDIA GPU or driver. On non-NVIDIA systems cudaGetDeviceCount()
returns 0 devices and llama.cpp falls back to CPU automatically.
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).parent
SRC_BIN = ROOT / "bin"
DST_BIN = ROOT / "bin_exe"

# Explicit list: core DLLs + executables
REQUIRED_FILES = [
    "llama-server.exe",
    "llama.dll",
    "ggml.dll",
    "ggml-base.dll",
    "ggml-cuda.dll",
    "ggml-rpc.dll",
    "mtmd.dll",
    "cublas64_13.dll",
    "cublasLt64_13.dll",
    "cudart64_13.dll",
    "libomp140.x86_64.dll",
]


def main():
    if not SRC_BIN.exists():
        print(f"[ERROR] Source bin/ not found: {SRC_BIN}")
        return 1

    if DST_BIN.exists():
        shutil.rmtree(DST_BIN)
    DST_BIN.mkdir(parents=True)

    copied = 0

    # Copy explicit files
    for name in REQUIRED_FILES:
        src = SRC_BIN / name
        if not src.exists():
            print(f"[ERROR] Required file missing: {src}")
            return 1
        shutil.copy2(src, DST_BIN / name)
        size_mb = src.stat().st_size / 1024 / 1024
        print(f"  copied  {name:<30} ({size_mb:>7.1f} MB)")
        copied += 1

    # Copy all ggml-cpu-*.dll variants (arch-specific, auto-selected at runtime)
    for src in sorted(SRC_BIN.glob("ggml-cpu-*.dll")):
        shutil.copy2(src, DST_BIN / src.name)
        size_mb = src.stat().st_size / 1024 / 1024
        print(f"  copied  {src.name:<30} ({size_mb:>7.1f} MB)")
        copied += 1

    total_mb = sum(f.stat().st_size for f in DST_BIN.glob("*")) / 1024 / 1024
    print(f"\n  {copied} files staged")
    print(f"  Total bin_exe size: {total_mb:.1f} MB")
    print(f"  Output: {DST_BIN}")
    return 0


if __name__ == "__main__":
    exit(main())
