# Changelog

All notable changes to Anita AI Voice Companion will be documented in this file.

## [Unreleased] - 2025-10-05

### Fixed

#### Audio Playback Issues
- Fixed Edge TTS compatibility by removing unsupported `output_format` parameter
- Changed TTS output from MP3 to WAV format for better browser compatibility
- Added WAV-based duration estimation for accurate lip-sync and expression timing
- Improved audio playback reliability in `tts_pipeline.py`

#### Real-time Voice Conversation
- **Frontend**: Integrated microphone streaming with WebSocket audio transmission
  - 48kHz → 16kHz downsampling with anti-aliasing
  - PCM16 chunk encoding and streaming via `audio_chunk` messages
  - Auto-resume AudioContext to handle browser autoplay policies
  - Toggle button for continuous listening mode
- **Backend**: Added server-side audio pipeline with connection-level queues
  - ASR background task per WebSocket connection
  - Unified response path: STT → LLM → TTS → frontend playback
  - State events for UI feedback: `listening | thinking | speaking`
  - Detailed logging: `[ASR]`, `[SPEECH]`, `[STT]`, `[LLM Latency]`, `[TTS Latency]`

#### Chinese Recognition & Response Quality
- Fixed Whisper STT defaulting to English recognition
- Forced Chinese language mode (`language="zh"`) for accurate transcription
- Added auto-detection support with language fallback option
- Resolved "unintelligent" responses caused by misrecognized input

#### Stability & User Experience
- Made audio pipeline optional via `ENABLE_AUDIO_PIPELINE` environment variable
- Implemented lazy loading: audio models load on first audio chunk (faster startup)
- Optimized Anthropic API availability check to avoid billable test requests
- Character name changed from "Ani" to "Anita"

#### Dependencies & Documentation
- Added missing dependencies: `edge-tts`, `anthropic` to `requirements.txt`
- Updated README to use `.env` file for API key configuration
- Added `.env.example` with audio pipeline toggle documentation
- Improved test script robustness in `run_all_tests.py`

### Technical Details

**Files Modified:**
- `tts_pipeline.py`: WAV output, duration estimation, removed `output_format` parameter
- `audio_pipeline.py`: Chinese language recognition, auto-detection support
- `main_full.py`: Audio pipeline lazy loading, WebSocket audio handling, state events
- `frontend/complete_v2.html`: Microphone streaming, downsampling, AudioContext management
- `llm_pipeline.py`: Non-billable availability check for Anthropic API
- `requirements.txt`: Added `edge-tts`, `anthropic`
- `README.md`: Updated configuration instructions
- `.env.example`: Added `ENABLE_AUDIO_PIPELINE` flag

**Commits:**
- `90d8f03`: Fix Edge TTS compatibility and API key configuration
- `324dc54`: Force Chinese language recognition in Whisper STT
- `0483e33`: Change character name from Ani to Anita

---

## [1.1.0] - 2025-10-02

### Added
- 3D character animation with VSeeFace integration
- VMC protocol for expression control
- Emotional expression system (joy, sad, anger, surprise, neutral)
- Professional UI redesign with gradient backgrounds
- Enhanced emotion detection system

### Security
- Moved API keys to environment variables
- Added `.env.example` template
- Updated `.gitignore` to exclude sensitive files

---

## [1.0.0] - 2025-10-01

### Added
- Initial release of Anita AI Voice Companion
- Bilingual support (Chinese + English)
- Voice cloning with XTTS-v2
- Local LLM via Ollama (Qwen2.5/Qwen3)
- OpenAI/Anthropic API integration
- Real-time WebSocket communication
- GPU-accelerated inference (CUDA support)
- Browser-based speech recognition
- FastAPI backend + HTML/JS frontend
## 2025-10-05

Added
- rag/ folder scaffolding for Retrieval-Augmented Generation (RAG)
  - rag/knowledge.py: lightweight dependency‑free retriever (EN tokens + CJK bigrams)
  - rag/data/: sample JSON (`sample_products.json`) and data conventions
  - rag/index/: reserved for future vector stores
  - rag/README.md: usage and structure
  - scripts/md_to_json.py: Markdown -> JSON converter (front‑matter or H2 sections)

Notes
- Not wired into the main pipeline yet. You can import `KnowledgeBase` and call `search()` where needed. Future changes can swap to a vector index without touching callers.
