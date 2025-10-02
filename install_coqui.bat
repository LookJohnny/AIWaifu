@echo off
echo ================================================
echo Installing Coqui TTS for Ani
echo ================================================
echo.

echo [1/3] Installing Coqui TTS...
python -m pip install TTS

echo.
echo [2/3] Creating voice_samples directory...
if not exist "voice_samples" mkdir voice_samples

echo.
echo [3/3] Testing installation...
python -c "from TTS.api import TTS; print('Coqui TTS installed successfully!')"

echo.
echo ================================================
echo Installation Complete!
echo ================================================
echo.
echo Next steps:
echo 1. (Optional) Add anime voice sample to: voice_samples\ani_voice.wav
echo 2. Edit main_full.py to enable Coqui TTS
echo 3. Restart Ani server
echo.
echo For detailed instructions, see: INSTALL_COQUI_TTS.md
echo.
pause
