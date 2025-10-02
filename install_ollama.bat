@echo off
echo ============================================================
echo Installing Ollama for Ani
echo ============================================================
echo.

echo Step 1: Downloading Ollama installer...
powershell -Command "& {Invoke-WebRequest -Uri 'https://ollama.com/download/OllamaSetup.exe' -OutFile '%TEMP%\OllamaSetup.exe'}"

if not exist "%TEMP%\OllamaSetup.exe" (
    echo [FAIL] Download failed!
    echo Please download manually from: https://ollama.com/download/windows
    pause
    exit /b 1
)

echo [OK] Downloaded!
echo.

echo Step 2: Running installer...
echo (This will open a window - click through the installer)
start /wait "%TEMP%\OllamaSetup.exe"

echo.
echo Step 3: Waiting for Ollama to start...
timeout /t 10 /nobreak

echo.
echo Step 4: Downloading Llama 3.1 8B model (~4.7GB)...
echo This will take 2-10 minutes depending on your internet speed
ollama pull llama3.1:8b

if errorlevel 1 (
    echo.
    echo [WARN] Ollama might not be installed yet.
    echo Please restart your terminal and run:
    echo   ollama pull llama3.1:8b
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [OK] Ollama installed successfully!
echo ============================================================
echo.
echo Next steps:
echo 1. Restart the Ani server (Ctrl+C and run: python main_full.py)
echo 2. Ani will now use your GPU for intelligent responses!
echo.
echo You should see: [OK] Ollama backend initialized
echo.
pause
