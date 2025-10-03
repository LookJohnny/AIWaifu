@echo off
echo Killing any existing servers...
taskkill /F /IM py.exe 2>nul

timeout /t 2 /nobreak >nul

echo Starting Ani server...
cd /d F:\Ani
set PYTHONIOENCODING=utf-8
py -u main_full.py
