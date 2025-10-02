@echo off
echo ========================================
echo   Starting Ani - AI Voice Companion
echo ========================================
echo.

REM Set UTF-8 encoding to support emoji
set PYTHONIOENCODING=utf-8

REM Start server
echo Starting server...
echo Open your browser to: http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.

python -u main_full.py

pause
