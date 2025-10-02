# 🤖 Ani - AI Voice Companion - Deployment Guide

Complete guide to deploy Ani on a fresh Windows PC using GitHub and Claude Code.

---

## 📋 **Prerequisites**

### **Required Software**
1. **Python 3.10+** - Download from [python.org](https://www.python.org/downloads/)
   - ✅ Check "Add Python to PATH" during installation
   - Verify: `python --version`

2. **Git** - Download from [git-scm.com](https://git-scm.com/)
   - Verify: `git --version`

3. **CUDA Toolkit** (for NVIDIA GPU acceleration)
   - Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   - Required for: RTX/GTX GPUs
   - Skip if using CPU-only mode

4. **Claude Code** (optional but recommended)
   - AI-powered coding assistant
   - Helps with debugging and customization

### **System Requirements**
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 30GB free space
- **GPU**: NVIDIA GPU with 4GB+ VRAM (optional, improves speed)
- **OS**: Windows 10/11

---

## 🚀 **Quick Start (5 Steps)**

### **Step 1: Clone Repository**
```bash
git clone https://github.com/LookJohnny/AIWaifu.git
cd AIWaifu
```

### **Step 2: Install Python Dependencies**
```bash
pip install -r requirements.txt
pip install TTS
```

This installs:
- FastAPI (web server)
- Ollama (local LLM)
- Coqui TTS (voice cloning)
- Faster-Whisper (speech recognition)

### **Step 3: Install Ollama**
Download and install Ollama from [ollama.com](https://ollama.com/)

Then download the bilingual AI model:
```bash
ollama pull qwen2.5:7b
```

This downloads a 4.7GB Chinese+English language model.

### **Step 4: Download Voice Sample** (Optional)
The project includes voice samples in `voice_samples/`:
- `wzy.wav` - Current default voice (1.1MB)
- `lin.wav` - Alternative voice (994KB)
- `wei.wav` - Another option

To use your own voice:
1. Record/download 5-10 seconds of clear speech
2. Save as `voice_samples/your_voice.wav`
3. Edit `main_full.py` line 127:
   ```python
   speaker_wav="voice_samples/your_voice.wav"
   ```

### **Step 5: Run Ani**
```bash
set PYTHONIOENCODING=utf-8
python -u main_full.py
```

**Windows PowerShell:**
```powershell
$env:PYTHONIOENCODING="utf-8"
python -u main_full.py
```

Open browser: **http://localhost:8000**

---

## 🔧 **Detailed Setup (If Quick Start Fails)**

### **A. Python Setup**
```bash
# Check Python version (must be 3.10+)
python --version

# Upgrade pip
python -m pip install --upgrade pip

# Install core dependencies
pip install fastapi uvicorn websockets
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install faster-whisper
pip install TTS
pip install aiohttp requests pydantic
```

### **B. CUDA Setup (GPU Acceleration)**
1. Check GPU compatibility:
   ```bash
   nvidia-smi
   ```
2. Install CUDA Toolkit 11.8 or 12.1
3. Verify PyTorch CUDA:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   Should print `True`

### **C. Ollama Setup**
```bash
# Install Ollama
# Download from: https://ollama.com/download

# Pull language model (choose one):
ollama pull qwen2.5:7b     # 4.7GB - Good for 8GB+ RAM
ollama pull qwen2.5:14b    # 9GB - Better quality, needs 16GB+ RAM

# Test Ollama
ollama run qwen2.5:7b "你好"
```

### **D. Voice Cloning Setup**
Coqui TTS will auto-download models on first run (~2GB).

**First launch takes 5-10 minutes** while downloading:
- XTTS-v2 model
- Silero VAD model
- Faster-Whisper model

---

## 📝 **Configuration**

### **Change AI Model**
Edit `main_full.py` line 95:
```python
model="qwen2.5:7b",  # Change to qwen2.5:14b, llama3.1:8b, etc.
```

### **Change Voice**
Edit `main_full.py` line 127:
```python
speaker_wav="voice_samples/wzy.wav"  # Change to your .wav file
```

### **Use OpenAI Instead of Ollama**
Edit `main_full.py` lines 101-107 (uncomment):
```python
llm_config = LLMConfig(
    backend="openai",
    model="gpt-4o-mini",
    openai_api_key="YOUR_API_KEY_HERE",
    character_name="Ani",
    character_personality="friendly and cheerful anime companion"
)
```

---

## 🐛 **Troubleshooting**

### **Issue: "qwen3:32b requires 16.5GB RAM"**
**Solution**: Use smaller model
```bash
ollama pull qwen2.5:7b
```
Edit `main_full.py` to use `qwen2.5:7b`

### **Issue: "'gbk' codec can't encode emoji"**
**Solution**: Run with UTF-8 encoding
```bash
set PYTHONIOENCODING=utf-8
python -u main_full.py
```

### **Issue: "CUDA out of memory"**
**Solutions**:
1. Close other GPU applications
2. Use smaller model (qwen2.5:7b instead of 14b)
3. Set `device="cpu"` in config (slower but works)

### **Issue: "Port 8000 already in use"**
**Solution**: Kill old processes
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Or change port in main_full.py:
uvicorn.run(app, host="0.0.0.0", port=8001)
```

### **Issue: Voice sounds robotic**
**Solution**:
1. Use longer voice sample (8-10 seconds)
2. Ensure sample is clear, no background noise
3. Try different voice samples

---

## 💡 **Using with Claude Code**

### **Prompt for Claude:**
```
I want to deploy the Ani AI Voice Companion project from GitHub on a new Windows PC.

Repository: https://github.com/LookJohnny/AIWaifu

Please help me:
1. Clone the repository
2. Install all dependencies (Python, Ollama, Coqui TTS)
3. Download the qwen2.5:7b model
4. Configure voice cloning with my voice sample
5. Run the server and test it

My system:
- Windows 11
- Python 3.11
- NVIDIA RTX 4060 (8GB VRAM)
- 16GB RAM

Walk me through each step and help debug any errors.
```

### **What Claude Will Do**:
1. ✅ Check your system requirements
2. ✅ Install missing dependencies
3. ✅ Clone the repository
4. ✅ Download AI models
5. ✅ Test each component
6. ✅ Fix any errors
7. ✅ Start the server

---

## 📦 **Project Structure**
```
AIWaifu/
├── main_full.py              # Main server (FastAPI)
├── llm_pipeline.py            # LLM backend (Ollama/OpenAI)
├── tts_pipeline.py            # Voice synthesis (Coqui TTS)
├── audio_pipeline.py          # Speech recognition (Whisper)
├── frontend/
│   └── index.html             # Web interface
├── voice_samples/
│   ├── wzy.wav                # Voice sample 1
│   ├── lin.wav                # Voice sample 2
│   └── wei.wav                # Voice sample 3
├── requirements.txt           # Python dependencies
└── DEPLOYMENT.md              # This file
```

---

## 🎯 **Next Steps After Deployment**

### **1. Customize Voice**
- Record your own voice (5-10 seconds clear speech)
- Replace `voice_samples/wzy.wav`
- Restart server

### **2. Improve Performance**
- Upgrade to larger model (qwen2.5:14b)
- Add more RAM
- Use better GPU

### **3. Add Features**
- Conversation memory
- Personality customization
- Mobile support
- Discord bot integration

### **4. Share Your Setup**
- Fork the repository
- Add your custom voice
- Share on GitHub

---

## 🆘 **Need Help?**

### **GitHub Issues**
Report bugs: https://github.com/LookJohnny/AIWaifu/issues

### **Claude Code**
Ask Claude to:
- Debug errors
- Add features
- Optimize performance
- Explain code

### **Community**
- Check existing issues
- Read documentation
- Ask questions

---

## ✅ **Verification Checklist**

After deployment, verify everything works:

- [ ] Python installed (`python --version`)
- [ ] Git installed (`git --version`)
- [ ] Repository cloned
- [ ] Dependencies installed (`pip list`)
- [ ] Ollama running (`ollama list`)
- [ ] Model downloaded (`qwen2.5:7b` appears)
- [ ] Server starts without errors
- [ ] Browser opens http://localhost:8000
- [ ] Microphone permission granted
- [ ] Can speak and get response
- [ ] Voice sounds like your sample
- [ ] Chinese and English both work

---

## 🎉 **Success!**

If all checks pass, your Ani AI Voice Companion is ready!

**Test it:**
1. Open http://localhost:8000
2. Click microphone button
3. Say: "你好" (Chinese) or "Hello" (English)
4. Ani responds with cloned voice

**Enjoy your AI companion! 🤖💬**
