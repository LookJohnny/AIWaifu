# 🎭 VSeeFace Setup Guide for Ani

Complete guide to set up 3D character animation with VSeeFace integration.

---

## 📥 Software Downloads

### 1. VRoid Studio (Character Creation)
- **URL**: https://vroid.com/en/studio
- **Size**: ~200MB
- **Platform**: Windows 10/11, macOS
- **Purpose**: Create custom 3D anime character
- **License**: Free

**Installation**:
1. Download from official website
2. Run installer
3. Follow installation wizard
4. Launch and verify it works

---

### 2. VSeeFace (Character Animation)
- **URL**: https://www.vseeface.icu/
- **Size**: ~150MB
- **Platform**: Windows only
- **Purpose**: Animate VRM character with VMC protocol
- **License**: Free

**Installation**:
1. Download latest version (v1.13.38+)
2. Extract ZIP to: `C:\Tools\VSeeFace\`
3. Right-click `VSeeFace.exe` → Properties → Unblock (if present)
4. Install .NET Framework if prompted
5. Launch `VSeeFace.exe` to verify

**Important**: Make sure Windows Firewall doesn't block VSeeFace!

---

### 3. OBS Studio (Display Capture)
- **URL**: https://obsproject.com/
- **Size**: ~100MB
- **Platform**: Windows/Mac/Linux
- **Purpose**: Capture VSeeFace and display on TV
- **License**: Free, open-source

**Installation**:
1. Download installer
2. Run installer with default settings
3. Launch OBS Studio
4. Skip auto-configuration wizard (we'll configure manually)

---

## 🎨 Part 1: Create Character in VRoid Studio

### Step 1: Start New Project
1. Launch VRoid Studio
2. Click **"New Character"**
3. Choose template:
   - **Female Preset** (recommended for Ani)
   - Or Male Preset
4. Save project as: `F:\Ani\character\ani_project.vroid`

### Step 2: Customize Face (20 min)

**Face Shape**:
- Face → Face Type
- Adjust width, chin, cheeks for anime style
- Keep it cute and friendly

**Eyes** (Most Important!):
- Face → Eyes
- Style: Large, rounded anime eyes
- Color: Purple or blue (AI theme)
- Size: 60-70% (larger than default)
- Enable eye highlights for sparkle

**Eyebrows**:
- Match hair color (you'll choose next)
- Thickness: Medium-thin
- Angle: Slight arch (friendly expression)

**Nose & Mouth**:
- Nose: Small, anime-style
- Mouth: Natural with slight smile

### Step 3: Customize Hair (20 min)

**Hairstyle**:
- Hair → Hair Presets
- Suggested: Long Straight or Long Wavy
- Or choose any style you like

**Hair Color** (Choose a theme):
- **Option 1**: Purple gradient (tech/AI aesthetic) ⭐ Recommended
- **Option 2**: Blue gradient (cool/calm vibe)
- **Option 3**: Pink gradient (cute/friendly vibe)

**Hair Physics**:
- Enable hair physics: ON
- This makes hair move naturally

### Step 4: Customize Body & Outfit (30 min)

**Body**:
- Body → Body Type
- Height: Medium
- Proportions: Anime standard

**Outfit Ideas**:
- **Option 1**: Hoodie + jeans (casual modern)
- **Option 2**: Tech outfit (futuristic)
- **Option 3**: School uniform (classic anime)

**Color Coordination**:
- Match outfit colors to hair theme
- Example: Purple hair → purple/black outfit

**Accessories** (Optional):
- Headphones (AI voice companion theme!)
- Tech gadgets
- Hair clips

### Step 5: Export Character

**Pre-export Check**:
- Rotate camera 360° - looks good from all angles?
- Test expression presets (smile, sad, etc.)
- No clipping issues?

**Export Settings**:
1. Click **"Camera/Exporter"** tab (top right)
2. Click **"Export"** button
3. Choose **"VRM Export"**
4. Configure settings:
   - **VRM version**: VRM 0.0 ⭐ (most compatible with VSeeFace)
   - **Texture size**: 2048x2048 (high quality)
   - **Reduce polygon count**: OFF
   - **Reduce blend shapes**: **OFF** ⚠️ (CRITICAL - needed for expressions!)
5. Character name: "Ani"
6. Author: Your name
7. Export to: `F:\Ani\character\ani_character.vrm`

**Wait**: Export takes 2-5 minutes

**Verify**:
- File exists: `ani_character.vrm`
- File size: 10-50MB (if > 50MB, re-export with 1024x1024 textures)

---

## 🎭 Part 2: Configure VSeeFace

### Step 1: Load Character
1. Launch **VSeeFace.exe**
2. Click **"Open VRM"** button (top left)
3. Browse to: `F:\Ani\character\ani_character.vrm`
4. Click **"Open"**
5. Wait 30-60 seconds for character to load
6. Character should appear in viewport!

### Step 2: Camera Settings
1. Click **"Camera"** button
2. Adjust camera position:
   - **Position**: Center character in frame
   - **Zoom**: Show head and upper body (torso to top of head)
   - **Angle**: Eye-level or slightly below
3. Click **"Save Camera 1"**

### Step 3: Visual Settings

**Background**:
- Settings → Background
- Choose: **Solid color (green)** for OBS chroma key
- Or: **Transparent** (if supported)

**Quality Settings**:
- Settings → Rendering
- **Anti-aliasing**: ON (4x or 8x)
- **Texture quality**: High
- **Shadow quality**: Medium-High
- **VSync**: ON
- **Frame rate limit**: 60 FPS

**Lighting**:
- Settings → Lighting
- **Light intensity**: 1.0-1.2 (bright)
- **Ambient light**: 0.3-0.5 (soft shadows)
- **Rim light**: ON (optional, highlights edges)

### Step 4: Test Expressions

**Manual Test**:
- Press keyboard keys **1-9**
- Each key triggers different expression
- **Write down which keys do what**:
  - Key ___ = Happy/Joy
  - Key ___ = Sad
  - Key ___ = Angry
  - Key ___ = Surprised
  - Key ___ = Neutral

### Step 5: Enable VMC Protocol ⭐ (CRITICAL)

**VMC Receiver Setup**:
1. Settings → **OSC/VMC**
2. Enable: **"VMC protocol receiver"** ✅
3. **Port**: 39539 (default, don't change!)
4. **IP**: 127.0.0.1 (localhost)
5. **Protocol**: OSC/UDP
6. Click **"Apply"**

**Verify Connection**:
- Status should show "Receiver ON" or green indicator
- Port 39539 should be listening
- If Windows Firewall prompts, click **"Allow"**

**Blend Shape Names** (Important for Python code):
1. Settings → Blend Shapes
2. **Write down exact names** for emotions:
   - Joy/Happy: ___________________
   - Sad/Sorrow: ___________________
   - Anger/Angry: ___________________
   - Surprise/Surprised: ___________________
   - Neutral/Relaxed: ___________________

**Note**: These names must match in `animation_controller.py`! If different, update the code.

### Step 6: Audio Settings (for Lip-sync)

**Enable Audio Lip-sync**:
1. Settings → Audio
2. Enable: **"Lip sync from microphone"** or **"Lip sync from system audio"**
3. Select audio device: **Desktop Audio** (system audio)
4. **Lip sync sensitivity**: 0.5-1.5 (adjust if too sensitive/not sensitive enough)

**Test Lip-sync**:
- Play music or speak
- Character's mouth should move
- Adjust sensitivity if needed

### Step 7: Idle Animations

**Enable Natural Movements**:
1. Settings → Idle Animation
2. **Automatic blinking**: ON
3. **Blink interval**: 3-5 seconds
4. **Breathing animation**: ON
5. **Head sway**: ON (optional, subtle)

**Test**: Watch character for 30 seconds - should blink and breathe naturally

### Step 8: Save Configuration

**Save Settings**:
- Click **"Save Settings"** button
- Configuration saved to VSeeFace default location
- Settings will load automatically next time

**Keep VSeeFace Running**: Don't close! Leave it running for Python integration.

---

## 🐍 Part 3: Test Python Integration

### Step 1: Test AnimationController Standalone

**Make sure**:
- VSeeFace is running with character loaded
- VMC receiver enabled (port 39539)

**Run test**:
```bash
cd F:\Ani
python animation_controller.py
```

**Expected Output**:
```
============================================================
Animation Controller Test
============================================================
Make sure VSeeFace is running with VMC receiver enabled!
Port: 39539

✅ Connected to VSeeFace

Testing: joy (intensity: 1.0)
Testing: sad (intensity: 0.8)
Testing: anger (intensity: 0.9)
Testing: surprise (intensity: 1.0)
Testing: neutral (intensity: 0.5)
Returning to neutral...

✅ Test complete!
```

**What to observe in VSeeFace**:
- Character expressions should change every 2 seconds
- Joy → Sad → Anger → Surprise → Neutral
- If expressions don't change: Check blend shape names in code!

### Step 2: Start Full Backend

**Start Server**:
```bash
cd F:\Ani
python main_full.py
```

**Expected Output**:
```
============================================================
Initializing Ani v0 - Complete Voice Companion
============================================================
[OK] LLM pipeline initialized (qwen2.5:7b)
[OK] TTS pipeline initialized (Coqui XTTS-v2)
[OK] Audio pipeline initialized
✅ Animation controller connected to VSeeFace
============================================================
[OK] Ani v0 Server Ready!
============================================================
```

**If you see**: `[INFO] VSeeFace not running - animations disabled`
- VSeeFace is not running or VMC not enabled
- Double-check Steps 1-5 in Part 2

### Step 3: Test Full Pipeline

**Open Browser**:
- Go to: http://localhost:8000
- UI should load

**Have Conversation**:
1. Click **"Start Listening"**
2. Say: **"Hello Ani, tell me something happy"**
3. Observe:
   - ✅ Speech recognized
   - ✅ Status: Listening → Thinking → Speaking
   - ✅ LLM generates response
   - ✅ **Character expression changes to happy** 🎭
   - ✅ TTS audio plays
   - ✅ **Character mouth moves (lip-sync)** 👄
   - ✅ Character returns to neutral after

**Test All Emotions**:
- Joy: "Tell me good news"
- Sad: "Tell me a sad story"
- Anger: "What makes you angry?"
- Surprise: "Tell me something surprising"

**Check Console Logs**:
```
[User] Tell me something happy
[Ani] I'm so excited to talk with you! Today is wonderful!
[Emote] joy (0.85)
🎭 Expression: joy (85%)
[LLM Latency] 3500ms
```

---

## 📺 Part 4: OBS Studio Setup (Display on TV)

### Step 1: Create Scene

**New Scene Collection**:
1. Launch OBS Studio
2. Scene Collection → New
3. Name: **"Ani Display"**

**Create Scene**:
1. In Scenes panel (bottom left), click **+**
2. Name: **"Ani Character"**
3. Click OK

**Set Canvas Resolution**:
1. Settings → Video
2. **Base (Canvas) Resolution**:
   - For 1080p TV: **1920x1080**
   - For 4K TV: **3840x2160**
3. **Output (Scaled) Resolution**: Same as base
4. **FPS**: 60 (or 30 if performance issues)
5. Click Apply

### Step 2: Add VSeeFace Window Capture

**Add Source**:
1. In Sources panel, click **+**
2. Select **"Window Capture"**
3. Name: **"VSeeFace Character"**
4. Click OK

**Configure**:
- **Window**: Select "VSeeFace.exe"
- **Capture method**: Windows 10/11 (1903+)
- **Multi-adapter Compatibility**: OFF
- **Capture cursor**: OFF
- Click OK

**Position Character**:
- Character should appear in OBS preview
- Drag corners to resize (hold Shift = keep aspect ratio)
- Center character nicely in frame

### Step 3: Add Background (Optional)

**Add Background Source**:
1. Click **+** in Sources
2. Choose:
   - **"Image"** for static background
   - **"Media Source"** for video background
   - **"Color Source"** for solid color
3. Name: **"Background"**
4. Browse to image/video or choose color
5. Click OK

**Layer Order**:
- Drag **"Background"** below **"VSeeFace Character"** in Sources list
- Background appears behind character

### Step 4: Add Chroma Key (If Using Green Screen)

**If VSeeFace background is green**:
1. Right-click **"VSeeFace Character"** source
2. Select **"Filters"**
3. Click **+** in Effect Filters
4. Add: **"Chroma Key"**
5. **Key Color Type**: Green
6. Adjust **Similarity** and **Smoothness** until background removed
7. Click Close

### Step 5: Add Effects & Polish

**Color Correction**:
1. Right-click "VSeeFace Character" → Filters
2. Add: **"Color Correction"**
3. Adjust:
   - **Brightness**: +5 to +10 (if too dark)
   - **Contrast**: +5 to +10 (makes character pop)
   - **Saturation**: +5 (enhance colors)

**Sharpen (Optional)**:
1. Add: **"Sharpen"**
2. **Sharpness**: 0.10-0.30 (subtle)

### Step 6: Configure Audio Output

**Audio Settings**:
1. Settings → Audio
2. **Desktop Audio Device**: Select your TV (via HDMI) or system default
3. **Sample Rate**: 48kHz
4. **Channels**: Stereo

### Step 7: Fullscreen to TV

**Test Fullscreen**:
1. Right-click **"Ani Character"** scene (in preview or Scenes list)
2. Select: **"Fullscreen Projector (Scene)"**
3. Choose: Your TV display (Display 2, Display 3, etc.)
4. Scene should fullscreen on TV!

**Exit Fullscreen**: Press ESC

**For Permanent Setup**:
- Start OBS on Windows startup
- Set fullscreen projector to auto-start
- Or manually start when needed

### Step 8: Save Scene

**Export Scene**:
1. Scene Collection → Export
2. Save to: `F:\Ani\config\obs_ani_scene.json`
3. Can import later if needed

---

## 🧪 Testing Checklist

### ✅ VRoid Studio
- [ ] Character created and looks good
- [ ] Exported as ani_character.vrm (10-50MB)
- [ ] VRM version 0.0 used
- [ ] Blend shapes NOT reduced

### ✅ VSeeFace
- [ ] Character loads successfully
- [ ] Manual expressions work (keyboard 1-9)
- [ ] VMC receiver enabled (port 39539)
- [ ] Lip-sync from audio works
- [ ] Idle animations (blinking, breathing) work
- [ ] Configuration saved

### ✅ Python Integration
- [ ] `python animation_controller.py` test passes
- [ ] All 5 emotions trigger correctly
- [ ] `python main_full.py` starts without errors
- [ ] Console shows "Animation controller connected"

### ✅ Full Pipeline
- [ ] Browser loads http://localhost:8000
- [ ] Conversation works (voice in → voice out)
- [ ] Emotions detected from LLM
- [ ] Character expressions change based on emotions
- [ ] Lip-sync works during TTS playback
- [ ] Character returns to neutral after response

### ✅ OBS Studio
- [ ] VSeeFace window captured
- [ ] Character visible in OBS preview
- [ ] Background added and looks good
- [ ] Chroma key working (if used)
- [ ] Fullscreen projector displays on TV
- [ ] Audio plays through TV speakers

---

## 🐛 Troubleshooting

### Issue: VSeeFace Won't Load Character

**Error**: "Failed to load VRM file"

**Solutions**:
- ✅ Ensure file size < 50MB
- ✅ Re-export with smaller texture size (1024x1024)
- ✅ Use VRM 0.0 format (not VRM 1.0)
- ✅ Check file isn't corrupted (re-export from VRoid)

---

### Issue: VMC Connection Failed

**Error**: Animation controller shows "Could not connect to VSeeFace"

**Solutions**:
- ✅ Verify VSeeFace is running
- ✅ Check VMC receiver is enabled (Settings → OSC/VMC)
- ✅ Verify port is 39539
- ✅ Check Windows Firewall isn't blocking
- ✅ Try restarting VSeeFace
- ✅ Try restarting Python backend

**Test Connection**:
```bash
netstat -an | findstr 39539
```
Should show: `UDP 127.0.0.1:39539`

---

### Issue: Expressions Don't Change

**Symptoms**: Character stays neutral, no expression changes

**Solutions**:
- ✅ Check blend shape names in VSeeFace (Settings → Blend Shapes)
- ✅ Update `animation_controller.py` if names don't match:
  ```python
  EMOTION_BLEND_SHAPES = {
      "joy": "Joy",  # Or "Happy" or whatever VSeeFace shows
      "sad": "Sorrow",  # Or "Sad"
      # etc.
  }
  ```
- ✅ Test manually in VSeeFace (press 1-9 keys)
- ✅ Check console for OSC errors

---

### Issue: Lip-sync Not Working

**Symptoms**: Mouth doesn't move during TTS playback

**Solutions**:
- ✅ Enable "Lip sync from system audio" in VSeeFace
- ✅ Verify VSeeFace is monitoring correct audio device
- ✅ Check TTS audio is playing through system (not headphones only)
- ✅ Adjust lip-sync sensitivity (0.5-1.5)
- ✅ Test with louder volume

---

### Issue: OBS Shows Black Screen

**Symptoms**: Window Capture is black

**Solutions**:
- ✅ Change capture method to "Windows 10/11 (1903+)"
- ✅ Try "Game Capture" instead of "Window Capture"
- ✅ Run OBS as Administrator
- ✅ Check VSeeFace is not minimized
- ✅ Disable GPU-based capture if issues persist

---

### Issue: Low FPS / Performance Issues

**Symptoms**: Choppy animations, low frame rate

**Solutions**:
- ✅ **VSeeFace**: Lower rendering quality (Settings → Rendering)
- ✅ **OBS**: Reduce canvas resolution to 1280x720
- ✅ **OBS**: Disable filters (chroma key, color correction)
- ✅ Close other GPU-intensive applications
- ✅ Check GPU usage (should be < 80%)

**Check Performance**:
- Task Manager → Performance → GPU
- Should see VSeeFace + OBS + Python using GPU
- If > 90% sustained: Lower quality settings

---

### Issue: Character Looks Bad on TV

**Symptoms**: Blurry, pixelated, or bad colors

**Solutions**:
- ✅ **VSeeFace**: Increase texture quality (Settings → Rendering)
- ✅ **VSeeFace**: Adjust lighting (Settings → Lighting)
- ✅ **OBS**: Add color correction filter
- ✅ **OBS**: Add sharpen filter (subtle, 0.1-0.2)
- ✅ Use higher resolution export (2048x2048 textures)
- ✅ Adjust TV settings (picture mode, brightness)

---

### Issue: Audio Delay / Out of Sync

**Symptoms**: Lip-sync is noticeably delayed

**Solutions**:
- ✅ **VSeeFace**: Adjust audio delay offset (Settings → Audio)
  - Try values: -100ms to +100ms
- ✅ Reduce audio buffer size in system settings
- ✅ Check for Bluetooth audio delay (use wired speakers)
- ✅ Test with different audio devices

---

## 📚 Advanced Tips

### Custom Blend Shapes

If VSeeFace blend shape names are different, update `animation_controller.py`:

```python
EMOTION_BLEND_SHAPES = {
    "joy": "Your_Happy_Name",  # Check in VSeeFace settings!
    "sad": "Your_Sad_Name",
    "anger": "Your_Angry_Name",
    "surprise": "Your_Surprised_Name",
    "neutral": "Your_Neutral_Name"
}
```

### Adjust Expression Intensity

In `animation_controller.py`, modify intensity scaling:

```python
# Make expressions more subtle
intensity = intensity * 0.7  # 70% of original

# Or make more exaggerated
intensity = min(1.0, intensity * 1.3)  # 130% (capped at 100%)
```

### Multiple Characters

To switch between characters:
1. Create multiple VRM files
2. In VSeeFace: Open different VRM
3. Python code works with any loaded character!

---

## 🎯 Quick Start Summary

**For Next Time**:
1. ✅ Launch VSeeFace → Load `ani_character.vrm`
2. ✅ Enable VMC receiver (port 39539)
3. ✅ Start Python backend: `python main_full.py`
4. ✅ Launch OBS → Fullscreen to TV
5. ✅ Open browser: http://localhost:8000
6. ✅ Start talking to Ani! 🎉

**Total Setup Time**:
- First time: 4-6 hours (character creation + setup)
- After setup: 5 minutes (just launch software)

---

## 📞 Need Help?

**Check These First**:
1. ✅ VSeeFace is running
2. ✅ VMC receiver enabled (green indicator)
3. ✅ Port 39539 not blocked
4. ✅ Python console for errors
5. ✅ Test with `python animation_controller.py`

**Console Logs Help Debug**:
- Look for: `🎭 Expression: joy (85%)`
- If not appearing: VMC connection issue
- If appearing but character not changing: Blend shape name mismatch

---

**Congratulations! You now have a fully animated 3D AI companion!** 🎉

Enjoy chatting with Ani! 💜
