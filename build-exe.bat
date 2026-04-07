@echo off
setlocal enabledelayedexpansion

title Gemma4 LLM - Build EXE
echo.
echo  ==============================
echo   Gemma4 LLM - Build Single EXE
echo  ==============================
echo.

:: Check venv exists
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] venv not found. Run start.bat first to create it.
    pause
    exit /b 1
)

:: Install PyInstaller + pywebview if missing
venv\Scripts\python.exe -c "import PyInstaller" 2>nul
if %errorlevel% neq 0 (
    echo [SETUP] Installing PyInstaller...
    venv\Scripts\pip.exe install pyinstaller
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install PyInstaller
        pause
        exit /b 1
    )
)

venv\Scripts\python.exe -c "import webview" 2>nul
if %errorlevel% neq 0 (
    echo [SETUP] Installing pywebview...
    venv\Scripts\pip.exe install pywebview
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install pywebview
        pause
        exit /b 1
    )
)

venv\Scripts\python.exe -c "from PIL import Image" 2>nul
if %errorlevel% neq 0 (
    echo [SETUP] Installing Pillow (for icon generation)...
    venv\Scripts\pip.exe install Pillow
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install Pillow
        pause
        exit /b 1
    )
)

:: Check required source files
if not exist "bin\llama-server.exe" (
    echo [ERROR] bin\llama-server.exe missing. Run start.bat and download the engine first.
    pause
    exit /b 1
)
if not exist "models\gemma-4-E2B-it-Q4_K_M.gguf" (
    echo [ERROR] models\gemma-4-E2B-it-Q4_K_M.gguf missing. Run start.bat and download E2B first.
    pause
    exit /b 1
)
if not exist "models\mmproj-F16.gguf" (
    echo [ERROR] models\mmproj-F16.gguf missing. Run start.bat and download E2B first.
    pause
    exit /b 1
)

echo [STEP 1/5] Generating app icon...
venv\Scripts\python.exe make-icon.py
if %errorlevel% neq 0 (
    echo [ERROR] Icon generation failed
    pause
    exit /b 1
)

echo.
echo [STEP 2/5] Staging bundled binaries (llama-server + CUDA runtime)...
venv\Scripts\python.exe prepare-exe-bin.py
if %errorlevel% neq 0 (
    echo [ERROR] Failed to prepare bin_exe
    pause
    exit /b 1
)

echo.
echo [STEP 3/5] Cleaning old build artifacts...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"

echo.
echo [STEP 4/5] Running PyInstaller (this takes several minutes - bundling ~4.6 GB)...
venv\Scripts\python.exe -m PyInstaller gemma4-llm.spec --noconfirm
if %errorlevel% neq 0 (
    echo [ERROR] PyInstaller build failed
    pause
    exit /b 1
)

echo.
echo [STEP 5/5] Cleaning up...
if exist "build" rmdir /s /q "build"
if exist "bin_exe" rmdir /s /q "bin_exe"

echo.
echo  ==============================
echo   BUILD COMPLETE
echo  ==============================
echo.
echo   Output: dist\Gemma4-LLM.exe
echo.
dir /b dist\Gemma4-LLM.exe 2>nul && for %%A in ("dist\Gemma4-LLM.exe") do echo   Size: %%~zA bytes
echo.
pause
