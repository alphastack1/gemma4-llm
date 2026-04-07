@echo off
setlocal enabledelayedexpansion

title Gemma4 LLM
echo.
echo  ==============================
echo   Gemma4 LLM - Local Chat
echo  ==============================
echo.

:: Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install Python 3.11+ from python.org
    pause
    exit /b 1
)

:: Check Python version
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER%

:: Create venv if missing
if not exist "venv\Scripts\python.exe" (
    echo.
    echo [SETUP] Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create venv
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
    echo.
    echo [SETUP] Installing dependencies...
    venv\Scripts\pip.exe install --upgrade pip >nul 2>&1
    venv\Scripts\pip.exe install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed
) else (
    echo [OK] Virtual environment found
)

echo.
echo [START] Launching Gemma4 LLM...
echo         Close this window to stop the server.
echo.

venv\Scripts\python.exe app.py

pause
