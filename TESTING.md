# Testing Guide for Ani v0

## Prerequisites

1. Python 3.8+ installed
2. Virtual environment created and activated
3. Dependencies installed: `pip install -r requirements.txt`

## Manual Testing Checklist

### Phase 1: JSON Validation & WebSocket

**Test 1: Start basic server**
```bash
python main.py
```

Expected output:
```
Starting Ani v0 Server...
WebSocket endpoint: ws://localhost:8000/ws
Health check: http://localhost:8000/health
Metrics: http://localhost:8000/metrics
INFO:     Started server process [PID]
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Test 2: Health check**
```bash
# In browser or curl
curl http://localhost:8000/health
```

Expected JSON response with:
- `status: "healthy"`
- Latency stats for all pipeline stages

**Test 3: JSON validation client**
```bash
# In new terminal (venv activated)
python test_client.py
```

Expected output:
- ✓ Test 1: Echo message works
- ✓ Test 2: Valid LLM response validates successfully
- ✗ Test 3-5: Invalid messages are rejected with clear errors

### Phase 2: Audio Pipeline (VAD + STT)

**Test 4: Start audio server**
```bash
python main_audio.py
```

Expected startup sequence:
```
Initializing audio pipeline...
Loading Silero VAD model...
✓ Silero VAD loaded in XXXms
Loading Faster-Whisper (base) model...
✓ Faster-Whisper loaded in XXXms (device: cpu|cuda)
✓ Audio pipeline initialized
✓ Audio pipeline ready
```

**Test 5: Audio pipeline standalone**
```bash
python audio_pipeline.py
```

Expected:
- Models load successfully
- Test audio generates (1s sine wave)
- VAD detects speech in chunks
- STT attempts transcription (may be gibberish for sine wave)
- Latencies are printed

**Test 6: Audio WebSocket client**
```bash
python test_audio_client.py
```

Expected output:
- Test 1: Silence → `is_speech: false`, low probability
- Test 2: Tone → `is_speech: true` (maybe), higher probability
- Test 3: 10 chunks with latency stats
  - Average VAD latency <150ms ✓ (target met)
  - Min/Max latencies printed
- Test 4: Mixed JSON/audio messages work

### Latency Targets & Validation

| Stage | Target | How to Measure |
|-------|--------|----------------|
| VAD | <150ms | `test_audio_client.py` Test 3 output |
| STT | <300ms | TBD (need real speech input) |
| LLM | <400ms | Phase 3 |
| TTS | <700ms | Phase 4 |
| Total | <1.2s | End-to-end after Phase 4 |

## Common Issues & Troubleshooting

### Issue: ModuleNotFoundError

**Symptom:** `ModuleNotFoundError: No module named 'fastapi'`

**Solution:**
```bash
# Make sure venv is activated
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Torch not found / CUDA errors

**Symptom:** `No module named 'torch'` or CUDA-related errors

**Solution:**
```bash
# CPU-only installation (faster to install, slower inference)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU installation (if you have NVIDIA GPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Silero VAD download fails

**Symptom:** `torch.hub.load` hangs or fails

**Solution:**
- Check internet connection
- Silero downloads models from GitHub (first run only)
- Models are cached in `~/.cache/torch/hub/`
- If it hangs, try running `python audio_pipeline.py` once to pre-download

### Issue: WebSocket connection refused

**Symptom:** Client can't connect to `ws://localhost:8000/ws`

**Solution:**
- Make sure server is running first
- Check firewall isn't blocking port 8000
- Try `http://localhost:8000/health` in browser first

## Automated Testing (TODO)

Future additions:
- [ ] Unit tests with pytest
- [ ] Integration tests for audio pipeline
- [ ] Latency benchmarking suite
- [ ] Load testing for WebSocket
- [ ] CI/CD with GitHub Actions

## Performance Benchmarks

### Hardware Tested

**System 1** (to be filled after testing):
- CPU: ?
- GPU: ?
- RAM: ?

Results:
- VAD latency: ? ms
- STT latency: ? ms
- Total latency: ? ms

### Optimization Tips

1. **Use GPU for inference** (if available)
   - Whisper: ~3-5x faster on GPU
   - Silero VAD: Minimal difference (already very fast)

2. **Adjust Whisper model size**
   - `tiny`: Fastest, lower quality (~100ms)
   - `base`: Good balance (~200ms) ← **current default**
   - `small`: Better quality (~400ms)
   - `medium`: Best quality (~800ms)

3. **Tune VAD parameters**
   - Smaller `chunk_duration_ms`: Lower latency, more CPU
   - Larger `min_silence_duration_ms`: Wait longer before ending speech
   - Adjust `vad_threshold`: 0.5 is default (0.0-1.0)

## Next Steps

After Phase 2 testing passes:
- [ ] Phase 3: LLM integration (Ollama + Llama)
- [ ] Phase 4: TTS + phoneme extraction
- [ ] Phase 5: Character rendering + lipsync
- [ ] Phase 6: State machine + barge-in
