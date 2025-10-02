@echo off
echo ========================================
echo   Ani - AI Voice Companion Setup
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.10+ first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [OK] Python found

REM Check Git
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git not found! Please install Git first.
    echo Download from: https://git-scm.com/
    pause
    exit /b 1
)
echo [OK] Git found

REM Check Ollama
ollama --version >nul 2>&1
if errorlevel 1 (
    echo [WARN] Ollama not found!
    echo Please install Ollama from: https://ollama.com/
    echo Then run: ollama pull qwen2.5:7b
    pause
    exit /b 1
)
echo [OK] Ollama found

echo.
echo Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install requirements
    pause
    exit /b 1
)
echo [OK] Requirements installed

echo.
echo Installing Coqui TTS...
pip install TTS
if errorlevel 1 (
    echo [ERROR] Failed to install TTS
    pause
    exit /b 1
)
echo [OK] TTS installed

echo.
echo Checking Ollama models...
ollama list | findstr qwen2.5:7b >nul
if errorlevel 1 (
    echo [WARN] qwen2.5:7b not found
    echo Downloading model (4.7GB)...
    ollama pull qwen2.5:7b
    if errorlevel 1 (
        echo [ERROR] Failed to download model
        pause
        exit /b 1
    )
)
echo [OK] Model ready

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo To run Ani:
echo   1. Run: start_ani.bat
echo   2. Open: http://localhost:8000
echo.
pause
