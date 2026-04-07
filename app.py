"""
Gemma4 LLM - Local Chat Server (with Vision)
Manages llama-server subprocess + serves chat UI
"""

import os
import sys
import json
import time
import shutil
import signal
import zipfile
import logging
import threading
import subprocess
import webbrowser
from pathlib import Path

import requests
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).parent.resolve()

# When frozen via PyInstaller (--onefile), bundled read-only assets
# live at sys._MEIPASS (a temp dir that exists only while the EXE runs).
# Writable user data (models) live next to the EXE so they persist.
if getattr(sys, "frozen", False):
    BUNDLE_DIR = Path(sys._MEIPASS)
    USER_DIR = Path(sys.executable).parent.resolve()
    BIN_DIR = BUNDLE_DIR / "bin"
    STATIC_DIR = BUNDLE_DIR / "static"
    MODELS_DIR = USER_DIR / "models"
else:
    BIN_DIR = APP_DIR / "bin"
    MODELS_DIR = APP_DIR / "models"
    STATIC_DIR = APP_DIR / "static"

LLAMA_SERVER_PORT = 8080
APP_PORT = 7860
LLAMA_THREADS = 4

# Gemma 4 models — standard quantizations, stock llama.cpp
MODELS = {
    "E2B": {
        "name": "Gemma 4 E2B",
        "file": "gemma-4-E2B-it-Q4_K_M.gguf",
        "mmproj": "mmproj-F16.gguf",
        "url": "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf",
        "mmproj_url": "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/mmproj-F16.gguf",
        "size_mb": 3110,
        "mmproj_size_mb": 986,
        "description": "5B params, 2.3B active. Fast with vision.",
        "bundled": False,
    },
    "E4B": {
        "name": "Gemma 4 E4B",
        "file": "gemma-4-e4b-it-Q4_K_M.gguf",
        "mmproj": "mmproj-gemma-4-e4b-it-f16.gguf",
        "url": "https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-e4b-it-Q4_K_M.gguf",
        "mmproj_url": "https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/mmproj-gemma-4-e4b-it-f16.gguf",
        "size_mb": 5340,
        "mmproj_size_mb": 990,
        "description": "8B params, 4B active. Best balance of quality and speed.",
        "bundled": False,
    },
    "E4B-Q8": {
        "name": "Gemma 4 E4B Q8",
        "file": "gemma-4-e4b-it-Q8_0.gguf",
        "mmproj": "mmproj-gemma-4-e4b-it-f16.gguf",
        "url": "https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-e4b-it-Q8_0.gguf",
        "mmproj_url": "https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/mmproj-gemma-4-e4b-it-f16.gguf",
        "size_mb": 8030,
        "mmproj_size_mb": 990,
        "description": "8B params, 4B active. Best quality (Q8 quantization).",
        "bundled": False,
    },
}

# Stock llama.cpp — no special fork needed for Gemma 4.
BINARY_RELEASE_TAG = "b8683"
BINARY_BASE = f"https://github.com/ggml-org/llama.cpp/releases/download/{BINARY_RELEASE_TAG}"

# Auto-detect CUDA version from driver
def _detect_cuda_tag():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            major = int(r.stdout.strip().split("\n")[0].split(".")[0])
            return "13.1" if major >= 560 else "12.4"
    except Exception:
        pass
    return "12.4"

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder="static")
CORS(app)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("gemma4")

llama_process = None
active_model = None
download_progress = {}  # key -> {percent, status, error}
last_heartbeat = 0.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_llama_server():
    for name in ["llama-server.exe", "llama-server"]:
        path = BIN_DIR / name
        if path.exists():
            return path
    return None


def is_binary_ready():
    if not find_llama_server():
        return False
    return len(list(BIN_DIR.glob("cublas*.dll"))) > 0


def get_installed_models():
    result = {}
    for key, info in MODELS.items():
        model_path = MODELS_DIR / info["file"]
        mmproj_path = MODELS_DIR / info["mmproj"]
        result[key] = {
            **info,
            "key": key,
            "installed": model_path.exists() and mmproj_path.exists(),
            "model_downloaded": model_path.exists(),
            "mmproj_downloaded": mmproj_path.exists(),
            "path": str(model_path),
        }
    return result


def download_file(url, dest_path, progress_key):
    download_progress[progress_key] = {"percent": 0, "status": "connecting", "error": None}
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_suffix(dest_path.suffix + ".part")

    try:
        log.info(f"Downloading {url}")
        resp = requests.get(url, stream=True, timeout=60, allow_redirects=True)
        if resp.status_code != 200:
            raise Exception(f"HTTP {resp.status_code} from {resp.url}")
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0

        download_progress[progress_key]["status"] = "downloading"
        log.info(f"  Size: {total / 1024 / 1024:.1f} MB")

        with open(part_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    download_progress[progress_key]["percent"] = int(downloaded * 100 / total)

        part_path.rename(dest_path)
        download_progress[progress_key] = {"percent": 100, "status": "done", "error": None}
        log.info(f"Download complete: {dest_path.name} ({downloaded / 1024 / 1024:.1f} MB)")

    except Exception as e:
        download_progress[progress_key] = {"percent": 0, "status": "error", "error": str(e)}
        log.error(f"Download failed ({progress_key}): {e}")
        if part_path.exists():
            part_path.unlink()


def download_and_extract_zip(url, dest_dir, progress_key):
    download_progress[progress_key] = {"percent": 0, "status": "connecting", "error": None}
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp_zip = dest_dir / "_download.zip.part"

    try:
        log.info(f"Downloading {url}")
        resp = requests.get(url, stream=True, timeout=120, allow_redirects=True)
        if resp.status_code != 200:
            raise Exception(f"HTTP {resp.status_code}")
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0

        download_progress[progress_key]["status"] = "downloading"
        log.info(f"  Size: {total / 1024 / 1024:.0f} MB")

        with open(tmp_zip, "wb") as f:
            for chunk in resp.iter_content(chunk_size=2 * 1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    download_progress[progress_key]["percent"] = int(downloaded * 85 / total)

        download_progress[progress_key]["status"] = "extracting"
        download_progress[progress_key]["percent"] = 90
        log.info("  Extracting...")

        with zipfile.ZipFile(tmp_zip) as zf:
            zf.extractall(dest_dir)

        tmp_zip.unlink(missing_ok=True)
        download_progress[progress_key] = {"percent": 100, "status": "done", "error": None}
        log.info(f"  Done: extracted to {dest_dir}")

    except Exception as e:
        download_progress[progress_key] = {"percent": 0, "status": "error", "error": str(e)}
        log.error(f"Download failed ({progress_key}): {e}")
        tmp_zip.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# llama-server management
# ---------------------------------------------------------------------------

def start_llama_server(model_key):
    global llama_process, active_model

    stop_llama_server()

    exe = find_llama_server()
    if not exe:
        log.error("llama-server not found in bin/")
        return False

    model_info = MODELS.get(model_key)
    if not model_info:
        log.error(f"Unknown model: {model_key}")
        return False

    model_path = MODELS_DIR / model_info["file"]
    mmproj_path = MODELS_DIR / model_info["mmproj"]

    if not model_path.exists():
        log.error(f"Model file not found: {model_path}")
        return False

    cmd = [
        str(exe),
        "-m", str(model_path),
        "--host", "127.0.0.1",
        "--port", str(LLAMA_SERVER_PORT),
        "-c", "8192",                    # Context length (reasonable default)
        "-ngl", "99",                    # Offload all layers to GPU
        "-t", str(LLAMA_THREADS),
        "--no-webui",
    ]

    # Add multimodal projector if available
    if mmproj_path.exists():
        cmd += ["--mmproj", str(mmproj_path)]
        log.info("  Vision enabled (mmproj loaded)")

    log.info(f"Starting llama-server with {model_info['name']}...")
    log.info(f"  Command: {' '.join(cmd)}")

    try:
        creation_flags = 0
        if sys.platform == "win32":
            import ctypes
            SEM_FAILCRITICALERRORS = 0x0001
            SEM_NOGPFAULTERRORBOX = 0x0002
            ctypes.windll.kernel32.SetErrorMode(
                SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX
            )
            creation_flags = subprocess.CREATE_NO_WINDOW

        llama_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(BIN_DIR),
            creationflags=creation_flags if sys.platform == "win32" else 0,
        )
        active_model = model_key

        for i in range(90):  # Up to 90 seconds (larger models take longer)
            time.sleep(1)
            try:
                r = requests.get(f"http://127.0.0.1:{LLAMA_SERVER_PORT}/health", timeout=2)
                if r.status_code == 200:
                    log.info(f"llama-server ready on port {LLAMA_SERVER_PORT}")
                    return True
            except requests.ConnectionError:
                pass

            if llama_process.poll() is not None:
                output = llama_process.stdout.read().decode(errors="replace")
                log.error(f"llama-server exited with code {llama_process.returncode}")
                log.error(f"Output: {output[-2000:]}")
                llama_process = None
                active_model = None
                return False

        log.error("llama-server did not become ready in 90s")
        stop_llama_server()
        return False

    except Exception as e:
        log.error(f"Failed to start llama-server: {e}")
        llama_process = None
        active_model = None
        return False


def stop_llama_server():
    global llama_process, active_model
    if llama_process:
        pid = llama_process.pid
        log.info(f"Stopping llama-server (PID {pid})...")
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True, timeout=10,
                )
            else:
                llama_process.terminate()
                llama_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            llama_process.kill()
        except Exception as e:
            log.warning(f"Cleanup warning: {e}")
        finally:
            llama_process = None
            active_model = None
            log.info("llama-server stopped.")


def is_llama_running():
    return llama_process is not None and llama_process.poll() is None


# ---------------------------------------------------------------------------
# Routes - Static
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


@app.route("/fonts/<path:filename>")
def font_files(filename):
    return send_from_directory(STATIC_DIR / "fonts", filename)


# ---------------------------------------------------------------------------
# Routes - Status & Setup
# ---------------------------------------------------------------------------

@app.route("/api/status")
def api_status():
    global last_heartbeat
    last_heartbeat = time.time()

    return jsonify({
        "binary_installed": is_binary_ready(),
        "models": get_installed_models(),
        "active_model": active_model,
        "llama_running": is_llama_running(),
        "downloads": download_progress,
        "platform": "cpu",
    })


@app.route("/api/goodbye", methods=["POST"])
def api_goodbye():
    log.info("Received goodbye from browser. Shutting down.")
    def _delayed_exit():
        time.sleep(0.5)
        stop_llama_server()
        os._exit(0)
    threading.Thread(target=_delayed_exit, daemon=True).start()
    return "", 204


@app.route("/api/setup/binary", methods=["POST"])
def api_setup_binary():
    if find_llama_server():
        return jsonify({"ok": True, "message": "Already installed"})

    if "binary" in download_progress and download_progress["binary"]["status"] in ("connecting", "downloading", "extracting"):
        return jsonify({"ok": True, "message": "Already downloading"})

    cuda_tag = _detect_cuda_tag()

    def do_download():
        # Step 1: llama.cpp binary
        bin_url = f"{BINARY_BASE}/llama-{BINARY_RELEASE_TAG}-bin-win-cuda-{cuda_tag}-x64.zip"
        download_and_extract_zip(bin_url, BIN_DIR, "binary")
        if download_progress["binary"]["status"] != "done":
            return

        # Step 2: CUDA runtime DLLs
        cudart_url = f"{BINARY_BASE}/cudart-llama-bin-win-cuda-{cuda_tag}-x64.zip"
        download_and_extract_zip(cudart_url, BIN_DIR, "cudart")

    threading.Thread(target=do_download, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/setup/model", methods=["POST"])
def api_setup_model():
    data = request.get_json() or {}
    model_key = data.get("model", "E2B")

    if model_key not in MODELS:
        return jsonify({"ok": False, "error": f"Unknown model: {model_key}"}), 400

    model_info = MODELS[model_key]

    # Check if already fully installed
    model_dest = MODELS_DIR / model_info["file"]
    mmproj_dest = MODELS_DIR / model_info["mmproj"]
    if model_dest.exists() and mmproj_dest.exists():
        return jsonify({"ok": True, "message": "Already installed"})

    progress_key = f"model_{model_key}"
    if progress_key in download_progress and download_progress[progress_key]["status"] == "downloading":
        return jsonify({"ok": True, "message": "Already downloading"})

    def do_download():
        # Download model GGUF
        if not model_dest.exists():
            download_file(model_info["url"], model_dest, progress_key)
            if download_progress[progress_key]["status"] != "done":
                return

        # Download mmproj for vision
        mmproj_key = f"mmproj_{model_key}"
        if not mmproj_dest.exists():
            download_file(model_info["mmproj_url"], mmproj_dest, mmproj_key)

    threading.Thread(target=do_download, daemon=True).start()
    return jsonify({"ok": True, "model": model_key})


@app.route("/api/setup/delete_model", methods=["POST"])
def api_delete_model():
    data = request.get_json() or {}
    model_key = data.get("model")

    if model_key not in MODELS:
        return jsonify({"ok": False, "error": "Unknown model"}), 400

    if model_key == active_model:
        stop_llama_server()

    model_info = MODELS[model_key]
    for fname in [model_info["file"], model_info["mmproj"]]:
        path = MODELS_DIR / fname
        if path.exists():
            path.unlink()

    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Routes - Model Loading
# ---------------------------------------------------------------------------

@app.route("/api/load", methods=["POST"])
def api_load_model():
    data = request.get_json() or {}
    model_key = data.get("model", "E2B")

    if model_key == active_model and is_llama_running():
        return jsonify({"ok": True, "message": "Already loaded"})

    success = start_llama_server(model_key)
    if success:
        return jsonify({"ok": True, "model": model_key})
    else:
        return jsonify({"ok": False, "error": "Failed to start llama-server. Check console for details."}), 500


@app.route("/api/unload", methods=["POST"])
def api_unload_model():
    stop_llama_server()
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Routes - Chat (proxy to llama-server)
# ---------------------------------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def api_chat():
    if not is_llama_running():
        return jsonify({"error": "No model loaded"}), 503

    data = request.get_json()
    data["stream"] = True

    try:
        resp = requests.post(
            f"http://127.0.0.1:{LLAMA_SERVER_PORT}/v1/chat/completions",
            json=data,
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()

        def generate():
            for line in resp.iter_lines():
                if line:
                    yield line.decode("utf-8") + "\n"

        return Response(
            stream_with_context(generate()),
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    except requests.ConnectionError:
        return jsonify({"error": "llama-server not responding"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Heartbeat watchdog
# ---------------------------------------------------------------------------

HEARTBEAT_TIMEOUT = 20

def watchdog():
    while last_heartbeat == 0.0:
        time.sleep(1)
    log.info("Browser connected.")

    while True:
        time.sleep(5)
        if time.time() - last_heartbeat > HEARTBEAT_TIMEOUT:
            log.info(f"No heartbeat for {HEARTBEAT_TIMEOUT}s — shutting down.")
            stop_llama_server()
            os._exit(0)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

import atexit

def cleanup(signum=None, frame=None):
    log.info("Cleanup triggered.")
    stop_llama_server()
    if signum is not None:
        sys.exit(0)

atexit.register(cleanup)
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)
if sys.platform == "win32":
    signal.signal(signal.SIGBREAK, cleanup)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def extract_bundled_model():
    """In frozen mode, copy bundled model to persistent MODELS_DIR."""
    if not getattr(sys, "frozen", False):
        return
    for name in ["gemma-4-E2B-it-Q4_K_M.gguf", "mmproj-F16.gguf"]:
        bundled = Path(sys._MEIPASS) / "models" / name
        target = MODELS_DIR / name
        if bundled.exists() and not target.exists():
            log.info(f"First run: extracting {name} to {target}")
            shutil.copy2(bundled, target)
            log.info(f"Extracted: {target.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    MODELS_DIR.mkdir(exist_ok=True)
    if not getattr(sys, "frozen", False):
        BIN_DIR.mkdir(exist_ok=True)
    extract_bundled_model()

    threading.Thread(target=watchdog, daemon=True).start()

    frozen = getattr(sys, "frozen", False)

    if frozen:
        def run_flask():
            app.run(host="127.0.0.1", port=APP_PORT, debug=False,
                    threaded=True, use_reloader=False)

        threading.Thread(target=run_flask, daemon=True).start()

        # Only auto-start model if already downloaded (models not bundled in EXE)
        e2b = MODELS["E2B"]
        if (MODELS_DIR / e2b["file"]).exists() and (MODELS_DIR / e2b["mmproj"]).exists():
            threading.Thread(
                target=lambda: start_llama_server("E2B"), daemon=True
            ).start()
        else:
            log.info("Model not found — setup UI will handle download.")

        for _ in range(50):
            try:
                requests.get(f"http://127.0.0.1:{APP_PORT}/", timeout=0.2)
                break
            except Exception:
                time.sleep(0.1)

        import webview
        log.info("Opening native window...")
        webview.create_window(
            "Gemma4 LLM",
            f"http://localhost:{APP_PORT}",
            width=1100, height=800,
            min_size=(600, 500),
        )
        webview.start()
        stop_llama_server()
        os._exit(0)
    else:
        def open_browser():
            time.sleep(1.5)
            webbrowser.open(f"http://localhost:{APP_PORT}")

        threading.Thread(target=open_browser, daemon=True).start()

        log.info(f"Gemma4 LLM starting on http://localhost:{APP_PORT}")
        app.run(host="127.0.0.1", port=APP_PORT, debug=False, threaded=True)
