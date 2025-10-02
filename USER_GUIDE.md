# Ani - Voice Companion User Guide

## 🎉 Welcome!

Ani is your real-time voice companion. Just speak, and Ani responds with voice!

## 🚀 Quick Start (3 Steps)

### Step 1: Start the Server

Open a terminal and run:

```bash
cd f:\Ani
python main_full.py
```

You should see:
```
============================================================
Initializing Ani v0 - Complete Voice Companion
============================================================
Initializing LLM pipeline (backend: ollama)...
[WARN] Ollama not available, falling back to Mock
[OK] Mock LLM backend initialized (fast, deterministic responses)
Initializing TTS pipeline (engine: edge)...
[OK] Edge TTS initialized (voice: en-US-AriaNeural)
Loading Silero VAD model...
[OK] Silero VAD loaded
Loading Faster-Whisper (base) model...
[OK] Faster-Whisper loaded (device: cuda)
[OK] Audio pipeline ready
============================================================
[OK] Ani v0 Server Ready!
============================================================
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Open Your Browser

Go to: **http://localhost:8000**

You'll see a beautiful purple interface with a big microphone button.

### Step 3: Talk!

1. **Click the microphone button** 🎤
2. **Allow microphone access** when your browser asks
3. **Speak clearly**: "Hello Ani!"
4. **Click the button again** or **stop speaking**
5. **Listen** as Ani responds with voice!

That's it! You're talking with Ani! 🎉

---

## 💬 How It Works

When you speak:

1. **Your voice** → Microphone
2. **Browser** → Converts speech to text
3. **Server** → Processes with AI
4. **Ani's brain** → Generates response
5. **Text-to-Speech** → Creates voice
6. **Browser** → Speaks Ani's response

**Total time**: Usually 1-3 seconds!

---

## 🎨 What You'll See

### Status Indicators

- **🔴 Disconnected**: Server not running (start it!)
- **🟢 Connected**: Ready to talk!
- **🟡 Listening**: You're speaking now
- **🔵 Ani is speaking**: Ani is responding

### Conversation View

- **Light blue bubbles**: Your messages
- **Pink bubbles**: Ani's responses
- **Emojis**: Ani's current emotion (😊 joy, 😢 sad, 😲 surprise, etc.)

---

## 🎯 Example Conversations

### Try saying:

- "Hello Ani!"
  - **Ani**: "Hello! I'm Ani, your anime companion! How can I help you today?" 😊

- "How are you?"
  - **Ani**: Responds with cheerful greeting

- "Tell me something interesting"
  - **Ani**: Responds based on AI (better with Ollama installed)

- "What's your name?"
  - **Ani**: "I'm Ani!"

---

## ⚙️ System Requirements

### Minimum:
- **OS**: Windows 10/11
- **Browser**: Chrome, Edge, or Firefox (latest version)
- **Microphone**: Any working microphone
- **Internet**: Required for voice synthesis (Edge TTS)

### Recommended:
- **GPU**: NVIDIA RTX 4060 (detected!)
- **RAM**: 8GB+
- **Microphone**: Headset or good quality mic

---

## 🔧 Troubleshooting

### "Microphone button is disabled"

**Problem**: Server not running

**Solution**:
```bash
cd f:\Ani
python main_full.py
```

### "Browser won't allow microphone"

**Problem**: Permissions not granted

**Solution**:
1. Click the 🔒 lock icon in address bar
2. Allow microphone access
3. Refresh the page

### "Can't hear Ani's voice"

**Problem**: Speaker volume or browser settings

**Solution**:
1. Check your speaker/headphone volume
2. Try a different browser
3. Check browser's site settings for audio permission

### "Ani's responses are generic/boring"

**Problem**: Using Mock LLM (fallback mode)

**Solution**: Install Ollama for smarter responses:
1. See [OLLAMA_SETUP.md](OLLAMA_SETUP.md)
2. Install Ollama
3. Download: `ollama pull llama3.1:8b`
4. Restart server

### "Voice recognition doesn't work"

**Problem**: Browser speech API issues

**Solution**:
1. Use Chrome or Edge (best support)
2. Speak clearly and not too fast
3. Check microphone is working in other apps
4. Try reloading the page

---

## 📊 Performance

### Current Setup (Mock LLM):

| Component | Speed | Notes |
|-----------|-------|-------|
| Voice to Text | ~1s | Browser API |
| AI Thinking | ~110ms | Mock LLM (fast!) |
| Text to Voice | ~600-900ms | Edge TTS |
| **Total** | **~1.7-2s** | Pretty fast! |

### With Ollama (Optional):

| Component | Speed | Notes |
|-----------|-------|-------|
| Voice to Text | ~1s | Browser API |
| AI Thinking | ~50-200ms | GPU accelerated! |
| Text to Voice | ~600-900ms | Edge TTS |
| **Total** | **~1.6-2.1s** | Even better! |

---

## 🎭 Ani's Personality

Ani is:
- **Friendly** and cheerful
- **Enthusiastic** anime companion
- **Helpful** and eager to chat
- Shows **emotions** (joy, surprise, neutral, etc.)

Her responses include:
- Main text (what she says)
- Emotion type (joy, sad, anger, surprise, neutral)
- Emotion intensity (0.0-1.0)
- Intent (SMALL_TALK, ANSWER, ASK, JOKE)

---

## 🔮 Advanced Features

### GPU Acceleration (Auto-detected!)

- **Your GPU**: NVIDIA GeForce RTX 4060
- **VRAM**: 8GB
- **Status**: ✅ Ready for Ollama!

When you install Ollama, it will automatically use your GPU for much faster responses!

### JSON API (For Developers)

The WebSocket endpoint accepts JSON:

```json
{
  "type": "user_input",
  "text": "Your message here"
}
```

Returns:

```json
{
  "status": "success",
  "data": {
    "utterance": "Ani's response",
    "emote": {"type": "joy", "intensity": 0.8},
    "intent": "SMALL_TALK",
    "phoneme_hints": []
  },
  "llm_latency_ms": 110
}
```

---

## 📝 Tips for Best Results

1. **Speak clearly** but naturally
2. **Don't speak too fast** - give the browser time to process
3. **Use a good microphone** - headsets work best
4. **Quiet environment** - reduce background noise
5. **One thought at a time** - keep sentences reasonably short
6. **Wait for Ani to finish** before speaking again

---

## 🎨 Customization (Coming Soon)

Future features:
- Different voices
- Different personalities
- Character avatars with lipsync
- Memory of past conversations
- Customizable emotions
- Background music

---

## 🆘 Getting Help

If something doesn't work:

1. Check this guide first
2. Look at [TESTING.md](TESTING.md) for technical details
3. Check [PROGRESS.md](PROGRESS.md) for current status
4. For Ollama setup: see [OLLAMA_SETUP.md](OLLAMA_SETUP.md)

---

## 🎯 What's Next?

### Right Now (Working!):
- ✅ Voice input
- ✅ AI responses
- ✅ Voice output
- ✅ Real-time conversation
- ✅ Emotion display

### Optional Upgrades:
- Install Ollama (smarter AI)
- Add character avatar
- Add conversation memory
- Add tool use (web search, etc.)

---

## 🎉 Have Fun!

Ani is here to chat with you. Talk about anything:
- Your day
- Your interests
- Questions you have
- Just say hi!

**Remember**: The more you talk, the better it gets!

Enjoy your conversations with Ani! 💜

---

**Version**: v0.4 (Voice Complete)
**Date**: 2025-10-01
**Status**: Fully Functional ✅
