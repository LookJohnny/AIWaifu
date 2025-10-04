# Ani - AI Voice Companion 🎤

A bilingual (Chinese + English) AI voice companion with 3D avatar animation, voice cloning, and real-time conversation.

## ✨ Features

- **🧠 Intelligent Conversation**: Powered by Claude 3.5 Haiku for natural, context-aware dialogue
- **🎤 Fast Voice Synthesis**: Edge TTS with multiple voice styles (<1s response time)
- **🗣️ Speech Recognition**: Real-time voice input with Faster-Whisper (GPU accelerated)
- **🎭 3D Avatar Animation**: VRM model with expression and lip-sync support
- **🌏 Bilingual Support**: Seamless Chinese and English conversation
- **⚡ Ultra-Fast Response**: <2s total latency (10x faster than baseline)

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- NVIDIA GPU (RTX 4060 or better recommended)
- Windows 10/11

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Ani.git
cd Ani
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API keys:
   - Edit `main_full.py` and add your Claude API key (line 98)

4. Start the server:
```bash
start_server.bat
```

5. Open browser:
```
http://localhost:8000
```

## 🎯 Voice Styles

Ani supports 5 different voice styles:

| Style | Voice ID | Description |
|-------|----------|-------------|
| **御姐** | zh-CN-XiaomoNeural | Mature, elegant |
| **萌萝莉** | zh-CN-XiaomengNeural | Cute, childlike |
| **熟女** | zh-CN-XiaohanNeural | Gentle, mature |
| **少妇** | zh-CN-XiaorouNeural | Warm, caring |
| **少女** | zh-CN-XiaoyiNeural | Lively, youthful ⭐ Default |

## 📦 Project Structure

```
Ani/
├── main_full.py              # Main server (FastAPI)
├── llm_pipeline.py           # LLM backends (Claude, OpenAI, Ollama)
├── tts_pipeline.py           # TTS engines (Edge TTS, Coqui)
├── audio_pipeline.py         # Voice activity detection & STT
├── animation_controller.py   # 3D avatar animation via VMC
├── frontend/
│   └── complete_v2.html      # Web UI with 3D avatar (professional design)
├── voice_samples/            # Voice cloning samples
└── character/                # VRM 3D models
```

## ⚙️ Configuration

### Switch Voice Style

Edit `main_full.py` line 143-148:

```python
tts_config = TTSConfig(
    engine="edge",
    voice="zh-CN-XiaoyiNeural",  # Change voice here
    rate="+8%",
    pitch="+5Hz"
)
```

### Switch LLM Backend

Edit `main_full.py` line 93-102:

```python
# Option 1: Claude (current)
llm_config = LLMConfig(
    backend="anthropic",
    model="claude-3-5-haiku-20241022",
    openai_api_key="your-api-key-here"
)

# Option 2: Local Ollama (free, no API key)
# llm_config = LLMConfig(
#     backend="ollama",
#     model="qwen2.5:7b"
# )
```

## 🎨 Tools Included

- `convert_voice_samples.py` - Convert MP4 to WAV with voice analysis
- `download_voice_samples.py` - Download Edge TTS voice samples
- `list_edge_voices.py` - List all available Edge TTS voices

## 🔧 Performance Metrics

| Component | Latency | Notes |
|-----------|---------|-------|
| LLM (Claude) | 0.5-1s | max_tokens=150 for speed |
| TTS (Edge) | <1s | 10x faster than Coqui |
| STT (Whisper) | <0.5s | GPU accelerated |
| **Total** | **<2s** | End-to-end response |

## 📝 TODO

### High Priority
- [ ] **Better Motion System**: More natural gestures and body animations
- [ ] **Auto Speak & Listen**: Voice activity detection (no need to hold button)
- [ ] **Easy Deployment**: One-click setup script with environment configuration
- [ ] **Auto-Test & Logging**: Automated testing with detailed error logs

### Future Enhancements
- [ ] Add conversation memory/context
- [ ] Support more languages (Japanese, Korean)
- [ ] Mobile app support
- [ ] Multi-character support

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - feel free to use this project for personal or commercial purposes.

## 🙏 Acknowledgments

- Claude API by Anthropic
- Edge TTS by Microsoft
- Faster-Whisper by OpenAI
- Three.js for 3D rendering
- VSeeFace for VMC protocol

---

**⚠️ Important**: Remember to remove API keys before committing to GitHub!

🤖 Built with [Claude Code](https://claude.com/claude-code)
