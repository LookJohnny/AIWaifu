Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Installing Coqui TTS for Ani" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/4] Installing Coqui TTS..." -ForegroundColor Yellow
python -m pip install TTS

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Coqui TTS installed successfully!" -ForegroundColor Green
} else {
    Write-Host "[FAIL] Installation failed. Please check your Python installation." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[2/4] Creating voice_samples directory..." -ForegroundColor Yellow
if (!(Test-Path "voice_samples")) {
    New-Item -ItemType Directory -Path "voice_samples" | Out-Null
    Write-Host "[OK] Directory created" -ForegroundColor Green
} else {
    Write-Host "[OK] Directory already exists" -ForegroundColor Green
}

Write-Host ""
Write-Host "[3/4] Testing Coqui TTS installation..." -ForegroundColor Yellow
python -c "from TTS.api import TTS; print('[OK] Coqui TTS is working!')"

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Coqui TTS test passed!" -ForegroundColor Green
} else {
    Write-Host "[WARN] Test failed, but installation may still work" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[4/4] Downloading XTTS-v2 model (this may take a few minutes)..." -ForegroundColor Yellow
Write-Host "Note: Model will be downloaded on first use if not done now" -ForegroundColor Gray
python -c "from TTS.api import TTS; print('Loading model...'); tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2'); print('[OK] Model ready!')"

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. (Optional) Add anime voice sample to: voice_samples\ani_voice.wav" -ForegroundColor White
Write-Host "2. Open main_full.py and change TTS config (see line ~103)" -ForegroundColor White
Write-Host "3. Change from:" -ForegroundColor White
Write-Host '   tts_config = TTSConfig(engine="edge", ...)' -ForegroundColor Gray
Write-Host "   To:" -ForegroundColor White
Write-Host '   tts_config = TTSConfig(engine="coqui", voice="tts_models/multilingual/multi-dataset/xtts_v2")' -ForegroundColor Gray
Write-Host "4. Restart Ani server" -ForegroundColor White
Write-Host ""
Write-Host "For detailed instructions, see: INSTALL_COQUI_TTS.md" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
