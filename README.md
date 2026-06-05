# Gemma4 LLM

A fully offline AI chat app powered by Google's **Gemma 4** with native multimodal vision. No cloud, no API keys, no accounts.

Ships with **Gemma 4 E2B** (5.1B params, 2.3B active) bundled, so it works the moment you open it. The new **Gemma 4 12B** dense model (Unsloth Dynamic 4-bit) can be downloaded from inside the app for higher quality. Gemma 4 is natively multimodal — it understands both text and images.

**Recent additions:**
- 🧵 **Automatic CPU-thread detection** (`cores-2`) instead of a hardcoded count — faster out of the box on both desktop and Android
- 🎛️ **Settings panel** — tune sampling (temperature, top-p, top-k, repeat penalty, max tokens) live, and engine params (threads, context size) with a reload
- ⬇️ **In-app model downloads** — grab Gemma 4 12B / E4B from the Models tab without leaving the app

## Pre-built downloads

If you just want to use the app, grab the pre-built binaries from HuggingFace:

| Platform | File | Size |
|----------|------|------|
| **Windows** | [Gemma4-LLM.zip](https://huggingface.co/alphastack1/gemma4-llm/resolve/main/Gemma4-LLM.zip) | 5.05 GB |
| **Android** | [gemma4-llm.apk](https://huggingface.co/alphastack1/gemma4-llm/resolve/main/gemma4-llm.apk) | 3.9 GB |

Everything is bundled -- model, inference engine, and UI. Nothing else to install.

---

## Building from source

This repo contains everything needed to reproduce both builds. The app uses stock [llama.cpp](https://github.com/ggml-org/llama.cpp) (release b8683) as the inference backend.

### Prerequisites

- **Python 3.11+** with pip
- **Git**
- Model files (downloaded automatically by the app on first run, or manually from [unsloth](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF)):
  - `gemma-4-E2B-it-Q4_K_M.gguf` (2.96 GB)
  - `mmproj-F16.gguf` (940 MB)

### Run the dev server

```bash
# 1. Clone the repo
git clone https://github.com/alphastack1/gemma4-llm.git
cd gemma4-llm

# 2. Launch (creates venv, installs deps, starts server)
start.bat
```

The app opens in your browser. On first run it downloads llama.cpp binaries and the model files automatically.

### Build the Windows zip

The Windows release is a portable zip containing `Gemma4-LLM.exe` + `models/` folder. The EXE bundles llama.cpp binaries (including CUDA runtime for GPU acceleration) and the chat UI via PyWebView.

**Requirements:** Windows 10+, Python 3.11+, ~10 GB free disk space

```bash
# 1. Run the dev server first (downloads llama.cpp binaries + model files)
start.bat

# 2. Build the EXE (generates icon, stages binaries, runs PyInstaller)
build-exe.bat

# 3. Package into zip (EXE + models side by side)
venv\Scripts\python.exe package-release.py
```

Output: `dist/Gemma4-LLM.zip` (~5 GB)

**What the build scripts do:**
- `make-icon.py` -- generates `gemma4.ico` (blue star icon)
- `prepare-exe-bin.py` -- copies the required llama.cpp DLLs/EXEs from `bin/` into `bin_exe/` for bundling
- `gemma4-llm.spec` -- PyInstaller spec that bundles the binaries, frontend, and icon into a single EXE
- `package-release.py` -- creates the final zip with the EXE and models folder

**Why a zip instead of a single EXE?** PyInstaller's `--onefile` mode has a hard 4 GB limit (32-bit archive offsets). The model files alone are ~3.9 GB, so they can't be bundled inside the EXE. Instead, the EXE looks for a `models/` folder next to itself at runtime.

### Build the Android APK

The Android app is a single APK that bundles everything -- the model is split into ~1 GB chunks (Android build tools can't handle >2 GB assets), then reassembled on first launch.

**Requirements:** Android Studio (with SDK 36 + NDK + CMake 4.1.2), ~20 GB free disk space, JDK 17

```bash
# 1. Place model files in the Android assets directory
#    The model must be split into ~1 GB chunks:
cd android/app/src/main/assets/

#    Split the main model:
split -b 1000m /path/to/gemma-4-E2B-it-Q4_K_M.gguf gemma-4-E2B-it-Q4_K_M.gguf.part_

#    Copy the vision encoder as-is (under 1 GB, no split needed):
cp /path/to/mmproj-F16.gguf .

# 2. Clone llama.cpp into the NDK build directory
cd ../cpp/
git clone --depth 1 --branch b8683 https://github.com/ggml-org/llama.cpp.git

# 3. Build the APK
cd ../../../../   # back to android/
./gradlew assembleRelease
```

Output: `android/app/build/outputs/apk/release/app-release.apk` (~4.1 GB)

**Key Android implementation details:**
- `MainActivity.java` -- reassembles the split model chunks on first launch, then starts the inference service
- `LlamaService.java` -- foreground service that runs `llama-server` as a native process
- `CMakeLists.txt` -- builds stock llama.cpp for arm64-v8a, renames the server binary to `libllama_server.so` for Android extraction
- `build.gradle` -- `aaptOptions { noCompress 'gguf', 'part_aa', 'part_ab', 'part_ac' }` prevents compression of large assets
- The APK uses `extractNativeLibs="true"` and `largeHeap="true"` for the large model

**Why split the model?** Android's build tools (aapt2) use Java arrays internally, which have a max size of ~2.1 GB. A single 2.96 GB file causes `"Required array size too large"`. Splitting into 3 x ~1 GB chunks avoids this limit.

## Project structure

```
gemma4-llm/
  app.py                  # Python backend (Flask + llama-server management)
  static/
    index.html            # Chat UI (single-file frontend)
    fonts/                # Outfit + JetBrains Mono
  requirements.txt        # Python dependencies
  start.bat               # Dev server launcher

  # Windows build
  build-exe.bat           # Main build script
  gemma4-llm.spec         # PyInstaller spec
  make-icon.py            # Icon generator
  prepare-exe-bin.py      # Binary staging for PyInstaller
  package-release.py      # Creates final zip

  # Android build
  android/
    app/
      build.gradle
      src/main/
        AndroidManifest.xml
        java/com/gemma4/llm/
          MainActivity.java
          LlamaService.java
        cpp/
          CMakeLists.txt    # Builds llama.cpp for Android
        res/                # Icons, strings, styles
```

## Models

| Model | Params | Quant | Notes |
|-------|--------|-------|-------|
| **Gemma 4 E2B** | 5.1B (2.3B active) | Q4_K_M | Bundled — runs out of the box, phone-friendly |
| **Gemma 4 12B** | 12B dense | UD-Q4_K_XL | Best quality; download from the Models tab (desktop / high-RAM) |

- Image encoder: `mmproj-F16.gguf` — part of Gemma 4's native vision architecture
- GGUFs from [unsloth](https://huggingface.co/unsloth) (Dynamic 2.0 quants); License: Apache 2.0

## Credits

- [Google](https://ai.google.dev/gemma) for the Gemma 4 model family
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) for the inference engine
- [unsloth](https://huggingface.co/unsloth) for GGUF quantizations
