# 🎭 Day 2: Multimodal 3D Animation Integration Plan

## 📋 **Context from Yesterday (Day 1)**

**Project**: Ani - AI Voice Companion
**Repository**: https://github.com/LookJohnny/AIWaifu
**Location**: F:\Ani\
**Current Server**: http://localhost:8000

### **What's Already Working** ✅
- ✅ Bilingual AI (Chinese + English) with qwen2.5:7b
- ✅ Voice cloning with XTTS-v2 (wzy.wav custom voice)
- ✅ Real-time voice conversation loop
- ✅ Professional web UI with status indicators
- ✅ Chat history display
- ✅ Emotional state detection (joy, sad, anger, surprise, neutral)
- ✅ FastAPI backend + WebSocket communication
- ✅ GPU acceleration (CUDA) for TTS and LLM

### **Current System Architecture**
```
User Voice → Browser Speech Recognition → WebSocket →
→ LLM (Ollama qwen2.5:7b) → Emotion Detection →
→ TTS (XTTS-v2) → Audio Output
```

---

## 🎯 **Today's Goal: Add 3D Visual Character**

### **What We're Building**
Integrate a 3D animated character that:
- Syncs lips to voice output (lip-sync)
- Displays emotions based on AI responses
- Shows idle animations (blinking, breathing)
- Runs smoothly on RTX 4060 GPU
- Uses FREE open-source tools only

---

## 🛠️ **Open-Source Technology Stack**

### **Option A: VRoid Studio + VSeeFace (RECOMMENDED - Easiest)**

**Why This Option:**
- ✅ FREE and beginner-friendly
- ✅ No coding for character creation
- ✅ Excellent lip-sync quality
- ✅ Anime-style characters
- ✅ Works perfectly with Windows
- ✅ GPU-accelerated
- ✅ Can output to OBS for TV display

**Software Stack:**
```
VRoid Studio (Free)
├─ Purpose: Create custom 3D anime character
├─ Platform: Windows/Mac
├─ Output: VRM file format
├─ Time: 1-2 hours to create character
└─ Link: https://vroid.com/en/studio

VSeeFace (Free)
├─ Purpose: Animate VRM character with AI
├─ Platform: Windows only
├─ Features:
│   ├─ Perfect lip-sync
│   ├─ Expression control via API
│   ├─ Idle animations built-in
│   ├─ Low CPU/GPU usage
│   └─ VMC protocol support
├─ Link: https://www.vseeface.icu/
└─ Python Integration: VMC protocol (OSC/UDP)

OBS Studio (Free)
├─ Purpose: Capture and display on TV
├─ Platform: Windows/Mac/Linux
└─ Link: https://obsproject.com/
```

**Architecture with VSeeFace:**
```
Python Backend
    ├─ Emotion Detection
    ├─ Audio Output (wzy.wav voice)
    └─ VMC Protocol Messages
            ↓
    VSeeFace Application
    ├─ Receives VMC commands
    ├─ Lip-sync from audio
    ├─ Expression changes
    └─ Idle animations
            ↓
    OBS Studio → HDMI → TV Display
```

---

### **Option B: Live2D Cubism + iFacialMocap (Anime 2D)**

**Why This Option:**
- ✅ Professional anime aesthetic
- ✅ More expressive than 3D
- ✅ Lower GPU requirements
- ⚠️ Requires some character setup

**Software Stack:**
```
Live2D Cubism (Free Viewer, $8/month for Editor)
├─ Purpose: Create 2D animated character
├─ Platform: Windows/Mac
├─ Output: .model3.json
└─ Link: https://www.live2d.com/

VTube Studio (Free)
├─ Purpose: Animate Live2D models
├─ Platform: Windows/Mac/iOS/Android
├─ Features:
│   ├─ Expression hotkeys
│   ├─ Python API available
│   └─ OBS integration
└─ Link: https://denchisoft.com/
```

---

### **Option C: Blender + Rhubarb Lip-Sync (Full Control)**

**Why This Option:**
- ✅ Maximum customization
- ✅ Professional-quality output
- ⚠️ Steeper learning curve
- ⚠️ More development time

**Software Stack:**
```
Blender (Free)
├─ Purpose: 3D character modeling & animation
├─ Platform: Windows/Mac/Linux
├─ Features: Everything
└─ Link: https://www.blender.org/

Rhubarb Lip Sync (Free)
├─ Purpose: Generate lip-sync from audio
├─ Platform: Command-line tool
├─ Output: Animation JSON
└─ Link: https://github.com/DanielSWolf/rhubarb-lip-sync

Python Integration
├─ bpy (Blender Python API)
└─ Real-time scene control
```

---

## 📅 **Tomorrow's Development Plan (Option A - VSeeFace)**

### **Phase 1: Setup (1-2 hours)**

**Task 1.1: Install Software**
```bash
Downloads needed:
1. VRoid Studio: https://vroid.com/en/studio
2. VSeeFace: https://www.vseeface.icu/
3. OBS Studio: https://obsproject.com/

Installation order:
1. VRoid Studio → Create character
2. VSeeFace → Test character
3. OBS Studio → Capture for TV
```

**Task 1.2: Create Basic Character in VRoid**
```
Character creation steps:
1. Open VRoid Studio
2. Start from preset template
3. Customize:
   ├─ Hair style & color
   ├─ Eye shape & color
   ├─ Face shape
   ├─ Body proportions
   ├─ Outfit/clothing
   └─ Accessories
4. Export as .vrm file
5. Save to: F:\Ani\character\ani_character.vrm

Time: 1-2 hours (can be refined later)
```

**Task 1.3: Test Character in VSeeFace**
```
VSeeFace setup:
1. Launch VSeeFace
2. Load ani_character.vrm
3. Test built-in features:
   ├─ Webcam tracking (for testing)
   ├─ Microphone lip-sync
   ├─ Expression hotkeys
   └─ Idle animations
4. Configure VMC protocol:
   ├─ Enable VMC receiver
   ├─ Port: 39539 (default)
   └─ Note the IP address
```

---

### **Phase 2: Python Integration (2-3 hours)**

**Task 2.1: Install VMC Protocol Library**
```bash
# Install python-osc for VMC communication
pip install python-osc
pip install asyncio
```

**Task 2.2: Create Animation Controller**
```python
# Claude will create: animation_controller.py

Features to implement:
├─ VMC protocol client
├─ Expression triggers:
│   ├─ joy → happy expression
│   ├─ sad → sad expression
│   ├─ anger → angry expression
│   ├─ surprise → surprised expression
│   └─ neutral → neutral expression
├─ Lip-sync trigger from audio playback
└─ Blinking and idle animation coordination

VMC Protocol Messages:
- /VMC/Ext/Blend/Val (blend shape values)
- /VMC/Ext/Hmd/Pos (head position)
- /VMC/Ext/Blend/Apply (apply changes)
```

**Task 2.3: Integrate with Existing System**
```python
# Update main_full.py to add animation controller

Integration points:
1. When LLM generates emotion:
   └─ Send expression command to VSeeFace

2. When TTS generates audio:
   ├─ Save audio file temporarily
   ├─ Send audio path to VSeeFace for lip-sync
   └─ Play audio synchronized

3. Idle state:
   └─ VSeeFace handles automatically (blinking, breathing)
```

---

### **Phase 3: OBS Integration (1 hour)**

**Task 3.1: Configure OBS Scene**
```
OBS Scene Setup:
1. Create new scene: "Ani Display"
2. Add sources:
   ├─ Window Capture → VSeeFace window
   ├─ Background image/video (optional)
   ├─ Audio output capture → System audio
   └─ Filters:
       ├─ Chroma Key (if using green screen)
       ├─ Color correction
       └─ Scaling/Transform
3. Set canvas resolution:
   ├─ 1920x1080 (Full HD) or
   └─ 3840x2160 (4K for TV)
4. Output: Fullscreen to TV via HDMI
```

**Task 3.2: Test Full Pipeline**
```
End-to-end test:
1. Start VSeeFace with character loaded
2. Start Python backend (main_full.py)
3. Start OBS with capture scene
4. Open browser to http://localhost:8000
5. Test conversation:
   User: "你好" (Chinese)
   ↓
   AI thinks (purple status)
   ↓
   AI responds with emotion (joy)
   ↓
   Character shows happy expression
   ↓
   Character lip-syncs to voice
   ↓
   Returns to idle state
```

---

### **Phase 4: Polish & Optimization (1-2 hours)**

**Task 4.1: Expression Mapping**
```python
# Refine emotion → expression mapping

Current emotions from LLM:
├─ joy (0.0-1.0 intensity)
├─ sad (0.0-1.0 intensity)
├─ anger (0.0-1.0 intensity)
├─ surprise (0.0-1.0 intensity)
└─ neutral (0.0-1.0 intensity)

VSeeFace expressions:
├─ Happy (blend value 0-1)
├─ Sad (blend value 0-1)
├─ Angry (blend value 0-1)
├─ Surprised (blend value 0-1)
├─ Neutral (blend value 0-1)
├─ Blinking (automatic)
└─ Mouth (lip-sync automatic)

Intensity mapping:
- Low intensity (0.0-0.5) → 50% expression blend
- High intensity (0.5-1.0) → 100% expression blend
```

**Task 4.2: Performance Tuning**
```python
Optimizations:
├─ Async expression updates (don't block TTS)
├─ Expression queue (smooth transitions)
├─ Debounce rapid expression changes
└─ Cache common expressions

Target performance:
├─ Expression change: <100ms
├─ Lip-sync delay: <50ms
├─ Total response: Still ~5-9 seconds
└─ FPS in VSeeFace: 60fps
```

**Task 4.3: Error Handling**
```python
Edge cases to handle:
├─ VSeeFace not running → Graceful degradation
├─ VMC connection lost → Auto-reconnect
├─ Expression conflicts → Priority system
├─ Audio sync issues → Adjust timing
└─ OBS crash → Restart integration
```

---

## 📝 **Detailed Task List for Claude Tomorrow**

### **Morning Session (2-3 hours)**

1. **Install and test VRoid Studio + VSeeFace**
   - Download software
   - Create basic character in VRoid
   - Export .vrm file
   - Load and test in VSeeFace
   - Configure VMC protocol

2. **Create animation_controller.py**
   ```python
   # Claude should create this file with:

   class AnimationController:
       def __init__(self, vmc_host="127.0.0.1", vmc_port=39539):
           """Initialize VMC OSC client"""

       async def set_expression(self, emotion: str, intensity: float):
           """Send expression command to VSeeFace"""

       async def trigger_lipsync(self, audio_file: str):
           """Trigger lip-sync for audio file"""

       async def reset_to_idle(self):
           """Return to neutral/idle state"""
   ```

3. **Integrate with main_full.py**
   - Add animation controller initialization
   - Hook into LLM emotion detection
   - Sync with TTS audio output
   - Test with simple conversation

---

### **Afternoon Session (2-3 hours)**

4. **Set up OBS Studio**
   - Install OBS
   - Create capture scene
   - Configure window capture for VSeeFace
   - Add background and effects
   - Set output to TV via HDMI

5. **Test full pipeline**
   - Run end-to-end conversation test
   - Verify expression changes
   - Check lip-sync quality
   - Confirm TV display working
   - Debug any issues

6. **Polish and optimize**
   - Refine expression intensity mapping
   - Smooth expression transitions
   - Optimize performance
   - Add error handling
   - Document the system

---

## 🎯 **Success Criteria for Tomorrow**

By end of day, you should have:

✅ **Visual Character**
- Custom anime character created in VRoid
- Character displays in VSeeFace
- Character visible on TV via OBS

✅ **Emotion Expressions**
- AI emotions trigger character expressions
- Smooth transitions between expressions
- Intensity levels working correctly

✅ **Lip-Sync**
- Character mouth syncs to voice output
- Minimal delay (<100ms)
- Works for both Chinese and English

✅ **Full System Integration**
- Voice input → AI response → Expression + Lip-sync
- All components working together
- No crashes or major bugs

✅ **Documentation**
- Code documented and clean
- Setup guide for VSeeFace integration
- Troubleshooting notes

---

## 📦 **Required Downloads**

Save these links for tomorrow:

1. **VRoid Studio**: https://vroid.com/en/studio
   - Size: ~200MB
   - Platform: Windows 10/11

2. **VSeeFace**: https://www.vseeface.icu/
   - Size: ~150MB
   - Platform: Windows only
   - Requires: .NET Framework

3. **OBS Studio**: https://obsproject.com/
   - Size: ~100MB
   - Platform: Windows/Mac/Linux

4. **Python OSC Library**:
   ```bash
   pip install python-osc
   ```

---

## 🚀 **Tomorrow's Prompt for Claude**

Copy and paste this to Claude Code tomorrow morning:

```
Hi Claude! Today I'm adding 3D visual character animation to my Ani AI Voice Companion.

**Current Project Status:**
- Repository: https://github.com/LookJohnny/AIWaifu
- Location: F:\Ani\
- Server: http://localhost:8000
- Working: Voice conversation with emotion detection

**Today's Goal:**
Integrate VSeeFace + VRoid Studio for 3D character animation with:
1. Lip-sync to TTS voice output (wzy.wav voice)
2. Expression changes based on AI emotions
3. Display on TV via OBS Studio
4. Full integration with existing FastAPI backend

**What I Need Your Help With:**

**Phase 1: Setup & Character Creation (Morning)**
1. Guide me through VRoid Studio character creation
2. Help me export and test in VSeeFace
3. Configure VMC protocol for Python control

**Phase 2: Python Integration (Afternoon)**
1. Create animation_controller.py for VMC communication
2. Integrate with main_full.py emotion detection
3. Sync lip-sync with TTS audio output
4. Handle expression transitions smoothly

**Phase 3: OBS & Testing (Evening)**
1. Set up OBS scene for TV display
2. Test full conversation pipeline
3. Debug and optimize performance
4. Document everything

**Tech Stack:**
- VRoid Studio (character creation)
- VSeeFace (animation + lip-sync)
- python-osc (VMC protocol)
- OBS Studio (TV display)
- Existing: FastAPI + qwen2.5:7b + XTTS-v2

**Current Emotion System:**
The LLM already outputs: joy, sad, anger, surprise, neutral (with intensity 0.0-1.0)
These need to map to VSeeFace blend shapes.

**System Specs:**
- Windows 11
- RTX 4060 8GB
- Python 3.11

Please guide me step-by-step. Write all necessary code, explain how it works, and help me debug any issues. Let's build this! 🚀
```

---

## 💡 **Alternative: Quick Prototype Path**

If VSeeFace setup is too complex, here's a simpler fallback:

### **Fallback Option: Ready Player Me + Three.js**

```javascript
// Web-based 3D character (runs in browser)

Stack:
├─ Ready Player Me (free 3D avatars)
├─ Three.js (3D rendering in browser)
├─ rhubarb-lip-sync (audio → visemes)
└─ Existing FastAPI backend

Advantages:
✓ No extra software installation
✓ Works in same browser as UI
✓ Cross-platform (works anywhere)
✓ Easier to deploy

Disadvantages:
✗ More coding required
✗ Less anime aesthetic
✗ May need performance optimization
```

Claude can implement this if VSeeFace path has issues.

---

## ✅ **Final Checklist for Tomorrow**

Before starting, make sure:
- [ ] Current server (b7e1c8) is still running properly
- [ ] GitHub is up to date
- [ ] You have ~4-6 hours available
- [ ] TV/monitor is ready for testing
- [ ] Headphones/speakers working
- [ ] Microphone working

During development:
- [ ] Take notes on what works/doesn't work
- [ ] Ask Claude to explain anything unclear
- [ ] Test frequently (don't wait until end)
- [ ] Save/commit progress regularly

---

## 🎉 **Expected Outcome**

By tomorrow evening, you'll have:

**Ani - Full Multimodal AI Companion**
- 🗣️ Voice conversation (already working)
- 🎭 3D animated character
- 😊 Emotional expressions
- 👄 Lip-sync to voice
- 📺 Beautiful display on TV
- 🎨 Custom anime character design
- 💾 All code on GitHub

**This will be AMAZING!** 🚀

Good luck tomorrow! Let's bring Ani to life visually! 💜
```

---

## 📊 **Time Estimate**

| Phase | Time | Difficulty |
|-------|------|------------|
| Software setup + character creation | 2-3 hours | Easy |
| Python VMC integration | 2-3 hours | Medium |
| OBS setup + testing | 1-2 hours | Easy |
| Polish + debug | 1-2 hours | Medium |
| **Total** | **6-10 hours** | **Medium** |

Perfect for a full day of development! 🎯
