# Ani - Open Source Anime Companion

Real-time, modular anime companion with <1.2s mic-to-voice latency.

## Current Status: Phase 2 Complete ✓

**Phase 1 - Foundation:**
- ✓ FastAPI WebSocket server with real-time communication
- ✓ Pydantic JSON schema validation (enforces strict LLM output format)
- ✓ Latency measurement system (tracks pipeline stages)
- ✓ Health check and metrics endpoints

**Phase 2 - Audio Pipeline:**
- ✓ Silero VAD integration (voice activity detection)
- ✓ Faster-Whisper STT integration (speech-to-text)
- ✓ WebSocket binary audio streaming
- ✓ Real-time VAD processing with latency tracking

## Quick Start

### Setup (Windows)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Server

**Option 1: JSON-only mode (Phase 1)**
```bash
python main.py
```

**Option 2: Full audio pipeline (Phase 2)**
```bash
python main_audio.py
```

Server will start on `http://localhost:8000`
- WebSocket: `ws://localhost:8000/ws`
- Health check: `http://localhost:8000/health`
- Metrics: `http://localhost:8000/metrics`

### Test

**Phase 1 - JSON validation:**
```bash
python test_client.py
```

**Phase 2 - Audio pipeline:**
```bash
# Test VAD and audio streaming
python test_audio_client.py

# Test audio pipeline standalone
python audio_pipeline.py
```

## JSON Contract (Strictly Enforced)

All LLM responses must match this schema:

```json
{
  "utterance": "string (required, max 500 chars)",
  "emote": {
    "type": "joy|sad|anger|surprise|neutral",
    "intensity": 0.0-1.0
  },
  "intent": "SMALL_TALK|ANSWER|ASK|JOKE|TOOL_USE",
  "phoneme_hints": [["phoneme", start_ms, end_ms], ...]
}
```

### Example Valid Response

```json
{
  "utterance": "Hello! I'm Ani, your anime companion!",
  "emote": {"type": "joy", "intensity": 0.8},
  "intent": "SMALL_TALK",
  "phoneme_hints": [
    ["HH", 0, 50],
    ["AH", 50, 150],
    ["L", 150, 200]
  ]
}
```

## Latency Targets

| Stage | Target | Current |
|-------|--------|---------|
| VAD | <150ms | TBD |
| STT | <300ms | TBD |
| LLM | <400ms | TBD |
| TTS | <700ms | TBD |
| **Total** | **<1.2s** | **TBD** |

JSON validation: ~1-5ms (measured in websocket echo test)

## Architecture

```
Audio Input → VAD → STT → LLM → TTS → Visemes → Character → Audio Output
                                ↓
                        JSON Schema Validator
                                ↓
                        Latency Metrics
```

## Pipeline Stages (v0 Roadmap)

### Phase 1: Foundation ✓ COMPLETE
- [x] FastAPI + WebSocket server
- [x] JSON schema validation
- [x] Latency tracking system
- [x] Health/metrics endpoints

### Phase 2: Audio Input (Next)
- [ ] Silero VAD integration
- [ ] faster-whisper STT streaming
- [ ] Audio chunk processing
- [ ] Measure VAD + STT latency

### Phase 3: LLM
- [ ] Ollama + Llama 3.1 8B setup
- [ ] Schema enforcement in prompts
- [ ] Fallback handling
- [ ] Measure LLM latency

### Phase 4: Audio Output
- [ ] Coqui TTS integration
- [ ] Phoneme extraction
- [ ] Phoneme → viseme mapping
- [ ] Measure TTS latency

### Phase 5: Character Rendering
- [ ] Three.js + VRM loader
- [ ] Basic character display
- [ ] Lipsync animation
- [ ] A/V sync validation

### Phase 6: State Management
- [ ] 3-state FSM (idle/listening/speaking)
- [ ] Barge-in support
- [ ] Graceful degradation
- [ ] Metrics dashboard

## Tech Stack

- **Backend**: FastAPI + WebSocket
- **Audio**: Silero VAD, Whisper, Coqui TTS (planned)
- **LLM**: Ollama + Llama 3.1 8B (planned)
- **Render**: Three.js + VRM (planned)
- **Validation**: Pydantic

## Project Structure

```
f:\Ani\
├── main.py                  # Phase 1: WebSocket server (JSON only)
├── main_audio.py            # Phase 2: Server with audio pipeline
├── audio_pipeline.py        # Phase 2: VAD + STT implementation
├── test_client.py           # Tests for Phase 1
├── test_audio_client.py     # Tests for Phase 2
├── validate.py              # Syntax validation helper
├── requirements.txt         # All Python dependencies
├── README.md                # This file
├── TESTING.md               # Comprehensive testing guide
├── PROGRESS.md              # Detailed development progress
├── .gitignore               # Git ignore patterns
├── setup.bat                # Windows setup script
└── run.bat                  # Windows run script
```

## Documentation

- **[README.md](README.md)** - Quick start and overview (this file)
- **[TESTING.md](TESTING.md)** - Complete testing guide with troubleshooting
- **[PROGRESS.md](PROGRESS.md)** - Detailed development log and milestones

## Development Principles

- Working code over perfect code
- Measure everything (latency at each stage)
- Graceful degradation (no crashes)
- Hot-swappable components
- Local/OSS first

## License

MIT (TBD - add license file)
