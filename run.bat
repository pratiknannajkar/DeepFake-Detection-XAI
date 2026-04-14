@echo off
title DeepShield AI - Starting...
color 0A

echo.
echo  ============================================
echo   DeepShield AI - Deepfake Detection Platform
echo  ============================================
echo.

:: Navigate to project root
cd /d "%~dp0"

:: Check Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo         Install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

:: Install dependencies only if fastapi is not already installed
echo [1/2] Checking dependencies...
python -c "import fastapi" >nul 2>nul
if %errorlevel% neq 0 goto INSTALL
goto SKIP_INSTALL

:INSTALL
echo       Installing required packages (first time only)...
pip install -r backend\requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo       Done!
goto START

:SKIP_INSTALL
echo       All dependencies already installed.

:START
echo.
echo [2/2] Starting DeepShield AI server...
echo.
echo  -----------------------------------------------
echo   Open your browser and go to:
echo   http://localhost:8000
echo  -----------------------------------------------
echo.
echo  Press Ctrl+C to stop the server.
echo.

title DeepShield AI - Running on http://localhost:8000
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

pause
