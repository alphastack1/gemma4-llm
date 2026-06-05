# Gemma4 LLM - Build Guide

> **Download ready-made builds:**
> - [**Windows**](https://huggingface.co/alphastack1/gemma4-llm/resolve/main/Gemma4-LLM.zip) (5.05 GB zip — extract, run EXE)
> - [**Android APK**](https://huggingface.co/alphastack1/gemma4-llm/resolve/main/gemma4-llm.apk) (3.9 GB)
> - [Release page](https://github.com/alphastack1/storage/releases/tag/gemma4-llm-v1.0.0)
>
> **Or build from source:** Drop this file into an empty folder, open it in
> Claude Code, and say *"Read this file and build everything."*

---

# Part 1: The Big Picture

```
 GEMMA4 LLM
 ================================================================

 A fully offline AI chat app with VISION. Zero cloud. Zero accounts.
 Runs Google's Gemma 4 (multimodal) entirely on your own device.

 Ships as two standalone packages:
 ┌─────────────────────────┐  ┌─────────────────────────┐
 │  Windows ZIP (5.05 GB)  │  │  Android APK (3.9 GB)   │
 │  Extract, run the EXE   │  │  Sideload to install     │
 │  Native window          │  │  Runs on any ARM64 phone │
 │  CUDA GPU + CPU         │  │  CPU inference           │
 └─────────────────────────┘  └─────────────────────────┘

 Both bundle Gemma 4 E2B + vision so they work the moment you open
 them. Bigger models (12B, E4B) download from inside the app.
```

## Architecture

```
 ┌─────────────────────────────────────────────────────────────┐
 │                     UI LAYER                                 │
 │  static/index.html  (single file)                           │
 │  Dark theme, streaming markdown, image upload (vision),      │
 │  chat history, + Settings panel (sampling & engine)          │
 │  No framework, no build step. Shared desktop + Android.      │
 └──────────────────────┬──────────────────────────────────────┘
                        │ HTTP on localhost
 ┌──────────────────────┴──────────────────────────────────────┐
 │                   SERVER LAYER                               │
 │  Desktop: app.py (Flask)     Android: LlamaService.java     │
 │  - Downloads engine/models   - Starts llama-server process  │
 │  - Manages subprocess        - Foreground service           │
 │  - Proxies SSE streams       - GemmaNative JS bridge        │
 └──────────────────────┬──────────────────────────────────────┘
                        │ subprocess
 ┌──────────────────────┴──────────────────────────────────────┐
 │                  INFERENCE ENGINE                            │
 │  llama-server (STOCK llama.cpp, release b9512)              │
 │  - OpenAI-compatible /v1/chat/completions endpoint          │
 │  - Multimodal via --mmproj (Gemma 4 native vision)          │
 │  - CUDA on desktop, CPU (arm64-v8a) on Android              │
 │  - Loads Gemma 4 *.gguf + mmproj-*.gguf                     │
 └─────────────────────────────────────────────────────────────┘
```

## Cost

```
 ┌──────────────┬────────────────────────────────────────────┐
 │ Everything   │ $0  (open-source models + engine)          │
 │ Disk space   │ 3.9 GB (APK) / 5 GB (Windows zip)         │
 │ Internet     │ Only to download extra models (12B/E4B)    │
 │ GPU          │ Optional. CPU works. NVIDIA GPU = faster.  │
 └──────────────┴────────────────────────────────────────────┘
```

---

# Part 2: Project Structure

```
 gemma4-llm/
 ├── app.py ·················· Flask backend + subprocess manager
 ├── requirements.txt ········ flask, flask-cors, requests
 ├── start.bat ··············· Dev launcher (creates venv, runs app)
 ├── static/
 │   ├── index.html ·········· ENTIRE frontend (chat + vision + settings)
 │   └── fonts/ ·············· Outfit + JetBrains Mono
 │
 │── EXE packaging:
 │   ├── gemma4-llm.spec ····· PyInstaller spec (EXE, model NOT bundled)
 │   ├── build-exe.bat ······· Build script
 │   ├── prepare-exe-bin.py ·· Stages binaries for bundling
 │   ├── make-icon.py ········ Generates gemma4.ico
 │   └── package-release.py ·· Zips EXE + models/ together
 │
 └── android/ ················ Android Studio project (com.gemma4.llm)
     └── app/src/main/
         ├── java/ ··········· MainActivity + LlamaService + GemmaNative
         ├── assets/ ········· index.html + bundled E2B (split parts)
         └── cpp/ ············ llama.cpp (stock, NDK arm64-v8a)

 Auto-created at runtime (gitignored):
 ├── venv/ ··················· Python virtual environment
 ├── bin/ ···················· llama-server.exe + DLLs
 └── models/ ················· Gemma-4-*.gguf + mmproj-*.gguf
```

---

# Part 3: The Models

```
 GEMMA 4 FAMILY (Google, Apache 2.0, GGUF via Unsloth)
 ================================================================

 ┌───────────────┬─────────────────────────────────┬────────┬───────────┐
 │ Model         │ File                            │ Size   │ Notes     │
 ├───────────────┼─────────────────────────────────┼────────┼───────────┤
 │ Gemma 4 E2B   │ gemma-4-E2B-it-Q4_K_M.gguf      │ 2.9 GB │ Bundled   │
 │ Gemma 4 E4B   │ gemma-4-e4b-it-Q4_K_M.gguf      │ ~5 GB  │ On demand │
 │ Gemma 4 12B   │ gemma-4-12b-it-UD-Q4_K_XL.gguf  │ 7.0 GB │ On demand │
 └───────────────┴─────────────────────────────────┴────────┴───────────┘

 + Vision projector: mmproj-F16.gguf (Gemma 4's native multimodal head)

 E2B = "Effective 2B" (5.1B params, 2.3B active) — small + fast, phone
 friendly, ships bundled so the app works instantly. The new 12B DENSE
 model is the quality flagship (~21 tok/s on an RTX 4060) — downloaded
 from the in-app Models tab. All GGUFs from huggingface.co/unsloth.
```

```
 LLAMA-SERVER LAUNCH FLAGS
 ================================================================

 llama-server
   -m models/gemma-4-E2B-it-Q4_K_M.gguf   model file
   --mmproj models/mmproj-F16.gguf         vision projector
   --host 127.0.0.1 --port 8080            localhost only
   -c 8192                                  context length (desktop)
   -ngl 99                                  all layers on GPU
   -t <auto>                                threads = CPU cores - 2
   --no-webui                               we have our own UI
```

---

# Part 4: The Backend (app.py)

```
 Three jobs:
 ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐
 │ 1. DOWNLOAD  │  │ 2. MANAGE    │  │ 3. PROXY             │
 │ Engine zips  │  │ Start/stop   │  │ Forward /api/chat    │
 │ Model GGUFs  │  │ llama-server │  │ to llama-server as   │
 │ + mmproj     │  │ subprocess   │  │ streaming SSE        │
 └──────────────┘  └──────────────┘  └──────────────────────┘
```

## API Routes

```
 ┌────────┬──────────────────────────┬──────────────────────────┐
 │ Method │ Path                     │ What                     │
 ├────────┼──────────────────────────┼──────────────────────────┤
 │ GET    │ /                        │ Serve index.html         │
 │ GET    │ /api/status              │ Full state + heartbeat   │
 │ POST   │ /api/setup/binary        │ Start engine download    │
 │ POST   │ /api/setup/model         │ Download model + mmproj  │
 │ POST   │ /api/load                │ Start llama-server       │
 │ POST   │ /api/unload              │ Stop llama-server        │
 │ GET    │ /api/settings            │ Engine params + auto info│
 │ POST   │ /api/reload              │ Apply threads/ctx, reload│
 │ POST   │ /api/chat                │ Stream chat completion   │
 └────────┴──────────────────────────┴──────────────────────────┘
```

## Engine Download

```
 Stock llama.cpp from ggml-org GitHub releases (tag b9512):
 ================================================================
 Step 1: Engine binary (llama-server.exe + ggml/mtmd DLLs)
 Step 2: CUDA runtime DLLs (cublas + cudart)

 CUDA version auto-detected from the NVIDIA driver:
   driver major >= 560  ──► CUDA 13.1 binaries
   driver major <  560  ──► CUDA 12.4 binaries
   no nvidia-smi (no GPU) ──► CUDA 12.4 (CPU fallback)
```

## NEW: Auto thread detection (the speed win)

```
 Old: hardcoded  -t 4   (left most of the CPU idle)
 New: -t = max(4, cpu_cores - 2)

 CPU inference scales ~linearly until memory bandwidth saturates
 around physical-core count, so cores-2 leaves headroom for the UI
 and system while using the rest. Auto on both desktop and Android.
 Users can override it in the Settings panel.
```

---

# Part 5: The Frontend (static/index.html)

```
 DESIGN SYSTEM
 ================================================================
 Theme:    Dark (zinc palette)
 Accent:   Blue (Gemma)
 Fonts:    Outfit (UI) + JetBrains Mono (code, tok/s)
 Vision:   Paperclip → attach an image → Gemma 4 sees it
```

## Side panel — three tabs

```
 ┌─────────────────────────────┐
 │ [Chats] [Models] [Settings] │
 │ ──────────────────────────  │
 │ CHATS:   history, new chat  │
 │ MODELS:  E2B (ok) / E4B /    │
 │          12B  [Download]     │
 │ SETTINGS:                    │
 │   Sampling (live):           │
 │     temperature, top_p,      │
 │     top_k, repeat_penalty,   │
 │     max_tokens               │
 │   Engine (reload):           │
 │     CPU threads (auto/N),    │
 │     context size             │
 └─────────────────────────────┘
```

## NEW: Settings panel

```
 Sampling params are sent per chat request (no reload).
 Engine params (threads, context) restart llama-server:
   Desktop ──► POST /api/reload
   Android ──► window.GemmaNative.applyEngineSettings(threads, ctx)
 Both persist to localStorage. Threads default to "Auto (cores-2)".
```

---

# Part 6: Packaging as a Windows ZIP

```
 Gemma4-LLM.zip  (~5 GB)
 ================================================================
 Contains:
   Gemma4-LLM/
     Gemma4-LLM.exe          (~917 MB: Python + Flask + llama.cpp
                              + CUDA runtime + UI, via PyInstaller)
     models/
       gemma-4-E2B-it-Q4_K_M.gguf   (2.9 GB)
       mmproj-F16.gguf              (940 MB)

 Double-click the EXE ──► native window (pywebview + Edge WebView2).
 The EXE looks for models/ NEXT TO itself — that's why it's a zip,
 not a bare exe.
```

```
 WHY A ZIP, NOT A SINGLE EXE?
 ================================================================
 PyInstaller --onefile re-extracts the WHOLE bundle to a temp dir
 on EVERY launch. A ~3 GB model inside would mean a multi-GB extract
 each time you open the app. Instead the model sits beside the EXE
 and loads directly — instant, no re-extract.
```

## Build

```
 1. start.bat                     # dev run: downloads engine + E2B
 2. build-exe.bat                 # make-icon → prepare-bin → PyInstaller
 3. python package-release.py     # zip: EXE + models/ side by side
 ──► dist/Gemma4-LLM.zip
```

---

# Part 7: Packaging as an Android APK

```
 gemma4-llm.apk  (~3.9 GB)  —  package com.gemma4.llm
 ================================================================
 Same UI + Gemma 4 E2B + vision, but no Python. The Flask layer is
 replaced by Java services; llama.cpp is compiled via the NDK.

 DESKTOP                         ANDROID
 ═══════                         ═══════
 Browser ──► Flask ──► llama     WebView ──► llama-server
             (Python)            LlamaService.java spawns it
                                 GemmaNative bridges settings
```

## Why the model is split

```
 Android's aapt2 uses Java arrays (max ~2.1 GB). A single 2.9 GB
 gguf fails with "Required array size too large", so E2B is split
 into ~1 GB chunks (.part_aa/ab/ac) in assets/ and reassembled to
 internal storage on first launch.
```

## Build

```
 cd android
 JAVA_HOME="…/Android Studio/jbr"
 ANDROID_HOME="…/Android/Sdk"
 ./gradlew.bat assembleRelease
 ──► app/build/outputs/apk/release/app-release.apk  (signed, sideload-ready)
```

---

# Part 8: What's New in v1.1.0

```
 ★ Gemma 4 12B — the new dense flagship, downloadable in-app
 ★ Auto CPU-thread detection (cores-2) — faster out of the box
 ★ Settings panel — live sampling + engine (threads/ctx) tuning
 ★ Both builds ship with E2B bundled — works the moment you open it
```

---

# Credits

- [Google](https://ai.google.dev/gemma) — Gemma 4 model family
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — inference engine
- [Unsloth](https://huggingface.co/unsloth) — Dynamic GGUF quantizations
