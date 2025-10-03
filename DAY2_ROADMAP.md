# 🎭 Day 2 Roadmap: 3D Character Animation Integration

**Date**: 2025-10-02
**Project**: Ani - AI Voice Companion
**Goal**: Add 3D animated character with lip-sync and emotional expressions

---

## 📊 Timeline Overview

```
Morning Session (3 hours)     → Setup & Character Creation
Afternoon Session (3 hours)   → Python Integration & Testing
Evening Session (2 hours)     → OBS Setup & Final Polish

Total Estimated Time: 8 hours
```

---

## 🌅 Morning Session (9:00 AM - 12:00 PM)

### Phase 1: Software Installation (30 minutes)
**Time**: 9:00 AM - 9:30 AM

- [ ] **Download VRoid Studio** (200MB)
  - URL: https://vroid.com/en/studio
  - Install to default location
  - Launch and verify it works

- [ ] **Download VSeeFace** (150MB)
  - URL: https://www.vseeface.icu/
  - Extract to: `C:\Tools\VSeeFace\`
  - Install .NET Framework if prompted
  - Launch and verify it works

- [ ] **Download OBS Studio** (100MB)
  - URL: https://obsproject.com/
  - Install to default location
  - Launch and verify it works

- [ ] **Install Python OSC Library**
  ```bash
  pip install python-osc
  ```

**Success Criteria**: All software launches without errors

---

### Phase 2: Character Creation in VRoid Studio (90 minutes)
**Time**: 9:30 AM - 11:00 AM

#### Step 1: Create New Character (10 min)
- [ ] Launch VRoid Studio
- [ ] Click "New Character"
- [ ] Choose template: "Female Preset" or "Male Preset"
- [ ] Save project as: `F:\Ani\character\ani_project.vroid`

#### Step 2: Customize Face (20 min)
- [ ] **Face Shape**: Adjust face width, chin, cheeks
- [ ] **Eyes**:
  - Style: Anime/rounded
  - Color: Your choice (suggest purple/blue for AI theme)
  - Size: Slightly larger for anime aesthetic
- [ ] **Eyebrows**: Match hair color, adjust thickness
- [ ] **Nose**: Small, anime-style
- [ ] **Mouth**: Natural shape, slight smile

#### Step 3: Customize Hair (20 min)
- [ ] **Hairstyle**: Choose from presets or customize
  - Suggest: Long hair with some volume
  - Color: Vibrant color (purple, blue, pink)
- [ ] **Hair Physics**: Enable for natural movement
- [ ] Add hair accessories (optional): Clips, ribbons, etc.

#### Step 4: Customize Body & Clothing (30 min)
- [ ] **Body Proportions**:
  - Height: Medium
  - Proportions: Anime standard
- [ ] **Outfit**: Choose or customize
  - Suggest: Casual modern or futuristic tech outfit
  - Colors: Match hair color theme
- [ ] **Accessories** (optional): Headphones, gadgets for AI theme

#### Step 5: Export Character (10 min)
- [ ] Click "Camera/Exporter" tab
- [ ] Select "Export" → "VRM Export"
- [ ] Settings:
  - Format: VRM 0.0 (most compatible)
  - Texture size: 2048x2048
  - Reduce blend shapes: OFF (need all expressions)
- [ ] Export to: `F:\Ani\character\ani_character.vrm`
- [ ] Verify file size (should be 10-50MB)

**Success Criteria**: ani_character.vrm file created and under 50MB

---

### Phase 3: VSeeFace Setup & Testing (60 minutes)
**Time**: 11:00 AM - 12:00 PM

#### Step 1: Load Character (10 min)
- [ ] Launch VSeeFace
- [ ] Click "Open VRM" button
- [ ] Navigate to: `F:\Ani\character\ani_character.vrm`
- [ ] Wait for character to load
- [ ] Verify character displays correctly

#### Step 2: Configure Basic Settings (15 min)
- [ ] **Camera Settings**:
  - Position: Center character in frame
  - Zoom: Head and upper body visible
  - Background: Transparent or solid color
- [ ] **Lighting**: Adjust for good visibility
- [ ] **Quality Settings**:
  - Anti-aliasing: ON
  - Texture quality: High
  - Frame rate: 60 FPS

#### Step 3: Test Built-in Features (15 min)
- [ ] **Test Webcam Tracking** (optional):
  - Enable webcam in settings
  - Verify face tracking works
  - Disable for now (will use VMC instead)
- [ ] **Test Microphone Lip-sync**:
  - Enable microphone input
  - Speak and verify mouth moves
  - Note the lip-sync quality
- [ ] **Test Expression Hotkeys**:
  - Press 1-9 for different expressions
  - Note which keys trigger which expressions

#### Step 4: Enable VMC Protocol (20 min)
- [ ] Click "Settings" → "VMC Protocol"
- [ ] **Enable VMC Receiver**:
  - Port: `39539` (default)
  - IP: `127.0.0.1` (localhost)
  - Protocol: OSC/UDP
- [ ] **Configure Blend Shapes**:
  - Enable all expression blend shapes
  - Map emotions:
    - Joy → Happy
    - Sad → Sad
    - Anger → Angry
    - Surprise → Surprised
    - Neutral → Neutral
- [ ] **Test VMC Connection**:
  - Keep VSeeFace running
  - Note: IP=127.0.0.1, Port=39539

**Success Criteria**:
- Character loads and displays correctly
- Expressions can be triggered manually
- VMC receiver is enabled and ready

---

## 🌤️ Afternoon Session (1:00 PM - 4:00 PM)

### Phase 4: Python Integration (90 minutes)
**Time**: 1:00 PM - 2:30 PM

#### Step 1: Create Animation Controller (30 min)
- [ ] Create file: `F:\Ani\animation_controller.py`
- [ ] Implement VMC OSC client
- [ ] Add expression mapping functions
- [ ] Add lip-sync trigger function
- [ ] Add error handling and reconnection logic

#### Step 2: Update Main Backend (30 min)
- [ ] Update `F:\Ani\main_full.py`
- [ ] Import AnimationController
- [ ] Initialize animation controller on startup
- [ ] Integrate with LLM emotion detection
- [ ] Sync with TTS audio output
- [ ] Add graceful degradation if VSeeFace not running

#### Step 3: Test Integration (30 min)
- [ ] Keep VSeeFace running with character loaded
- [ ] Start Python backend: `python main_full.py`
- [ ] Open browser: http://localhost:8000
- [ ] Test simple conversation:
  - Say: "Hello" (English)
  - Verify emotion detected
  - Verify expression changes in VSeeFace
  - Verify lip-sync works
- [ ] Test Chinese:
  - Say: "你好"
  - Verify same pipeline works
- [ ] Debug any issues

**Success Criteria**:
- AnimationController connects to VSeeFace
- Emotions trigger correct expressions
- Lip-sync works for both languages

---

### Phase 5: Full Pipeline Testing (90 minutes)
**Time**: 2:30 PM - 4:00 PM

#### Step 1: Multi-turn Conversation Test (30 min)
- [ ] Test various emotional responses:
  - Joy: "告诉我一个好消息" (Tell me good news)
  - Sad: "讲一个悲伤的故事" (Tell a sad story)
  - Surprise: "给我一个惊喜" (Give me a surprise)
  - Anger: "说说让人生气的事" (Talk about something annoying)
- [ ] Verify smooth expression transitions
- [ ] Check for any lag or delays
- [ ] Monitor system performance

#### Step 2: Edge Case Testing (30 min)
- [ ] Test rapid emotion changes
- [ ] Test long responses (30+ seconds)
- [ ] Test multiple emotions in one response
- [ ] Test neutral/no emotion responses
- [ ] Test interrupting during speech

#### Step 3: Performance Optimization (30 min)
- [ ] Monitor GPU usage (Task Manager)
- [ ] Monitor CPU usage
- [ ] Check response time metrics
- [ ] Optimize if needed:
  - Async expression updates
  - Expression debouncing
  - Audio sync timing adjustments

**Success Criteria**:
- All emotions display correctly
- Lip-sync is smooth (< 100ms delay)
- No crashes or freezes
- System runs stable for 10+ minutes

---

## 🌆 Evening Session (6:00 PM - 8:00 PM)

### Phase 6: OBS Studio Setup (60 minutes)
**Time**: 6:00 PM - 7:00 PM

#### Step 1: Create Scene (15 min)
- [ ] Launch OBS Studio
- [ ] Create new Scene Collection: "Ani Display"
- [ ] Create Scene: "Character View"
- [ ] Set canvas resolution:
  - 1920x1080 (Full HD) recommended
  - Or match your TV resolution

#### Step 2: Add Sources (20 min)
- [ ] **Add Window Capture**:
  - Source name: "VSeeFace Character"
  - Window: Select VSeeFace window
  - Capture method: Windows 10/11
- [ ] **Add Background** (optional):
  - Image or video
  - Place below character layer
  - Suggested: Gradient or abstract animation
- [ ] **Position Character**:
  - Scale to fill frame nicely
  - Center or offset as desired
  - Leave room for chat UI overlay (optional)

#### Step 3: Add Effects & Filters (15 min)
- [ ] **On Character Source**:
  - Chroma Key (if using green screen background in VSeeFace)
  - Color Correction (adjust brightness/contrast)
  - Sharpen filter (slight, for clarity)
- [ ] **On Scene**:
  - Add text overlay for "Ani" branding (optional)
  - Add audio visualizer (optional)

#### Step 4: Configure Output (10 min)
- [ ] Settings → Output:
  - Mode: Simple
  - Output: Fullscreen Projector
- [ ] Audio Settings:
  - Desktop Audio: Capture system audio
  - Verify audio levels
- [ ] Test fullscreen on TV:
  - Right-click scene → Fullscreen Projector → Select TV display

**Success Criteria**:
- Character visible in OBS
- Can fullscreen to TV via HDMI
- Audio plays through TV speakers

---

### Phase 7: Final Testing & Polish (60 minutes)
**Time**: 7:00 PM - 8:00 PM

#### Step 1: End-to-End Integration Test (20 min)
- [ ] Setup complete pipeline:
  1. VSeeFace running with character
  2. Python backend running (main_full.py)
  3. OBS capturing and displaying on TV
  4. Browser UI open on laptop
- [ ] Run full conversation test:
  - Speak to Ani in Chinese
  - Watch character on TV
  - Verify expressions + lip-sync
  - Check audio quality on TV speakers
- [ ] Record metrics:
  - Total response time
  - Expression change delay
  - Lip-sync delay
  - Any lag or stuttering

#### Step 2: Polish & Refinement (20 min)
- [ ] **Adjust Expression Intensity**:
  - If expressions too subtle: increase blend values
  - If too exaggerated: decrease blend values
- [ ] **Fine-tune Lip-sync Timing**:
  - Adjust audio delay if needed
  - Test with different sentence lengths
- [ ] **Optimize Visual Appearance**:
  - Lighting in VSeeFace
  - Camera angle and zoom
  - OBS filters and effects
- [ ] **Audio Balancing**:
  - Adjust TTS volume
  - Test on TV speakers
  - Ensure clear pronunciation

#### Step 3: Documentation & Cleanup (20 min)
- [ ] **Update README.md**:
  - Add 3D character animation section
  - Document VSeeFace setup steps
  - Add troubleshooting tips
- [ ] **Create Setup Guide**:
  - Save VSeeFace configuration
  - Export OBS scene collection
  - Document VMC protocol settings
- [ ] **Git Commit**:
  - Add new animation files
  - Commit with message: "Add 3D character animation with VSeeFace integration"
  - Push to GitHub

**Success Criteria**:
- Full pipeline works smoothly
- Character looks good on TV
- Everything documented
- Code committed to GitHub

---

## 📈 Success Metrics

By end of day, you should achieve:

### ✅ Functional Requirements
- [x] 3D character created and exported
- [x] Character loads in VSeeFace
- [x] VMC protocol connection working
- [x] Emotions trigger correct expressions
- [x] Lip-sync works for Chinese + English
- [x] Display working on TV via OBS
- [x] Full conversation pipeline functional

### ✅ Performance Requirements
- [x] Expression change delay: < 200ms
- [x] Lip-sync delay: < 100ms
- [x] Total response time: Still 5-9 seconds (unchanged)
- [x] VSeeFace FPS: 60fps stable
- [x] No crashes during 15-minute conversation
- [x] GPU usage: < 80% average

### ✅ Quality Requirements
- [x] Expressions clearly visible
- [x] Lip-sync looks natural
- [x] Smooth transitions between emotions
- [x] Character appearance is appealing
- [x] Audio quality maintained
- [x] TV display looks professional

---

## 🔧 Troubleshooting Guide

### Issue: VSeeFace Won't Load Character
**Solutions**:
- Ensure .vrm file is < 50MB
- Re-export from VRoid with reduced texture size
- Check .NET Framework is installed
- Try older VRM format (0.0 instead of 1.0)

### Issue: VMC Connection Fails
**Solutions**:
- Verify port 39539 is not blocked by firewall
- Check VSeeFace VMC receiver is enabled
- Restart both VSeeFace and Python backend
- Use `127.0.0.1` instead of `localhost`

### Issue: Lip-sync Delayed or Incorrect
**Solutions**:
- Enable audio input in VSeeFace settings
- Adjust audio delay offset in VSeeFace
- Check system audio routing
- Reduce audio buffer size in TTS config

### Issue: Expressions Not Changing
**Solutions**:
- Verify blend shape names in VMC settings
- Check emotion detection is working (console logs)
- Increase expression blend intensity values
- Test with manual VMC messages first

### Issue: OBS Performance Issues
**Solutions**:
- Reduce OBS canvas resolution to 1280x720
- Disable filters/effects
- Use Game Capture instead of Window Capture
- Close other GPU-intensive applications

### Issue: Character Looks Bad on TV
**Solutions**:
- Adjust VSeeFace lighting settings
- Improve camera angle and zoom
- Add color correction in OBS
- Use better background image/video

---

## 📦 Deliverables Checklist

### Code Files
- [ ] `animation_controller.py` - VMC protocol client
- [ ] `main_full.py` - Updated with animation integration
- [ ] `requirements.txt` - Updated with python-osc
- [ ] `config.json` - VSeeFace connection settings

### Character Assets
- [ ] `character/ani_project.vroid` - VRoid Studio project file
- [ ] `character/ani_character.vrm` - Exported VRM character
- [ ] `character/preview.png` - Character preview screenshot

### Configuration Files
- [ ] VSeeFace settings exported
- [ ] OBS scene collection exported
- [ ] VMC protocol configuration documented

### Documentation
- [ ] README.md updated with Day 2 features
- [ ] SETUP_VSEFACE.md - VSeeFace setup guide
- [ ] TROUBLESHOOTING.md - Common issues & solutions
- [ ] DAY2_ROADMAP.md - This roadmap document

### Testing
- [ ] All test cases passed (see TEST_CASES.md)
- [ ] Performance benchmarks recorded
- [ ] Edge cases documented

---

## 🎯 Next Steps (Day 3 Preview)

After completing Day 2, potential enhancements:

1. **Advanced Expressions**
   - Blending multiple emotions
   - Dynamic expression intensity
   - Custom expression animations

2. **Improved Lip-sync**
   - Phoneme-based mouth shapes
   - Chinese character pronunciation mapping
   - Better audio-visual sync

3. **Character Interactions**
   - Hand gestures
   - Head movements
   - Eye tracking user position

4. **Multiple Characters**
   - Different personalities
   - Character switching
   - Multi-character conversations

5. **Web Integration**
   - Embed character in browser
   - Three.js fallback option
   - Mobile support

---

## ✨ Motivation

**Remember**: By end of today, Ani will go from a voice-only AI to a **fully animated multimodal companion**!

You'll be able to:
- 🗣️ Have natural conversations in Chinese & English
- 😊 See emotions expressed visually
- 👄 Watch perfect lip-sync to the voice
- 📺 Display on your TV like a virtual assistant
- 🎨 Showcase your custom anime character

**This is HUGE progress!** Stay focused, take breaks, and enjoy the process! 🚀

---

**Last Updated**: 2025-10-02
**Status**: Ready to Execute
**Estimated Completion**: 8:00 PM (8 hours total)
