# ✅ Day 2 Master Checklist: 3D Character Animation Integration

**Date**: 2025-10-02
**Project**: Ani - AI Voice Companion
**Goal**: Add 3D animated character with expressions and lip-sync

---

## 🎯 Quick Start Checklist

Before you begin:
- [ ] Current Ani server is working (test at http://localhost:8000)
- [ ] Have 8 hours available today
- [ ] TV/monitor connected via HDMI
- [ ] Headphones and microphone working
- [ ] Stable internet for downloads
- [ ] F:\Ani\ folder is current working directory

---

## 📥 Pre-Installation Checklist

### System Requirements
- [ ] Windows 10/11 64-bit
- [ ] RTX 4060 GPU (8GB VRAM) - **Confirmed ✓**
- [ ] At least 20GB free disk space
- [ ] .NET Framework 4.7.2+ installed
- [ ] Python 3.11+ installed - **Confirmed ✓**
- [ ] Git installed - **Confirmed ✓**

### Software Downloads (Total: ~450MB)
- [ ] **VRoid Studio** (~200MB)
  - URL: https://vroid.com/en/studio
  - Version: Latest stable
  - Save to: Downloads folder

- [ ] **VSeeFace** (~150MB)
  - URL: https://www.vseeface.icu/
  - Version: Latest stable (v1.13.38+)
  - Extract to: `C:\Tools\VSeeFace\`

- [ ] **OBS Studio** (~100MB)
  - URL: https://obsproject.com/
  - Version: Latest stable (v30.0+)
  - Install to: Default location

---

## 🌅 MORNING SESSION (9:00 AM - 12:00 PM)

### ☑️ Phase 1: Software Installation (30 min)

#### VRoid Studio Installation
- [ ] Download completed
- [ ] Run installer
- [ ] Accept license agreement
- [ ] Install to default location
- [ ] Launch VRoid Studio
- [ ] Verify it opens without errors
- [ ] Close VRoid Studio

#### VSeeFace Installation
- [ ] Download completed
- [ ] Extract ZIP to `C:\Tools\VSeeFace\`
- [ ] Check folder contains: `VSeeFace.exe`
- [ ] Right-click VSeeFace.exe → Properties → Unblock (if present)
- [ ] Double-click `VSeeFace.exe` to launch
- [ ] If .NET error: Install .NET Framework from prompt
- [ ] Verify VSeeFace opens without errors
- [ ] Close VSeeFace

#### OBS Studio Installation
- [ ] Download completed
- [ ] Run installer
- [ ] Accept license agreement
- [ ] Install to default location
- [ ] Launch OBS Studio
- [ ] Skip auto-configuration wizard (we'll configure manually)
- [ ] Verify it opens without errors
- [ ] Close OBS Studio

#### Python Dependencies
- [ ] Open PowerShell in F:\Ani\
- [ ] Run: `pip install python-osc`
- [ ] Verify installation: `pip show python-osc`
- [ ] Should show version 1.8.0+
- [ ] No errors displayed

**Checkpoint**: All software installed and launches successfully ✓

---

### ☑️ Phase 2: Character Creation in VRoid (90 min)

#### Project Setup (5 min)
- [ ] Launch VRoid Studio
- [ ] Click "New Character"
- [ ] Choose template: Female Preset (or Male if preferred)
- [ ] Project name: "ani_project"
- [ ] Save location: `F:\Ani\character\` (create folder if needed)
- [ ] Click "Create"

#### Face Customization (25 min)

**Face Shape**
- [ ] Navigate to: Face → Face Type
- [ ] Adjust face width: Medium (slider ~50%)
- [ ] Adjust chin: Slightly rounded
- [ ] Adjust cheeks: Soft, anime-style
- [ ] Preview from multiple angles

**Eyes**
- [ ] Navigate to: Face → Eyes
- [ ] Eye style: Anime/large rounded style
- [ ] Eye color: Choose vibrant color (suggest purple/blue for AI theme)
- [ ] Eye size: Slightly larger than default (~60%)
- [ ] Eye position: Standard
- [ ] Eyelashes: Medium length
- [ ] Eye highlights: ON (for sparkle effect)

**Eyebrows**
- [ ] Navigate to: Face → Eyebrows
- [ ] Match color to hair (will set next)
- [ ] Thickness: Medium-thin
- [ ] Angle: Slightly arched (friendly look)

**Nose & Mouth**
- [ ] Navigate to: Face → Nose
- [ ] Size: Small (anime-style)
- [ ] Navigate to: Face → Mouth
- [ ] Shape: Natural with slight smile
- [ ] Lip color: Natural pink/red

**Preview**
- [ ] Rotate camera 360° to check face
- [ ] Test expression presets (joy, sad, anger)
- [ ] Verify eyes look good from all angles

#### Hair Customization (25 min)

**Hairstyle Selection**
- [ ] Navigate to: Hair → Hair Presets
- [ ] Choose style: Long hair (or your preference)
- [ ] Suggested: "Long Straight" or "Long Wavy"

**Hair Color**
- [ ] Navigate to: Hair → Hair Color
- [ ] Choose vibrant color:
  - Option 1: Purple gradient (tech/AI aesthetic)
  - Option 2: Blue gradient (cool/calm vibe)
  - Option 3: Pink gradient (cute/friendly vibe)
- [ ] Add highlights (optional): Lighter shade of base color

**Hair Details**
- [ ] Adjust hair length: Medium-long
- [ ] Adjust volume: Medium-high (for personality)
- [ ] Enable hair physics: ON
- [ ] Add accessories (optional):
  - Hair clips
  - Ribbons
  - Headband

**Preview**
- [ ] Test hair physics by moving character
- [ ] Verify hair doesn't clip through body
- [ ] Check hair from all angles

#### Body & Outfit Customization (30 min)

**Body Proportions**
- [ ] Navigate to: Body → Body Type
- [ ] Height: Medium (slider ~50%)
- [ ] Body proportions: Anime standard
- [ ] Adjust if needed:
  - Shoulder width: Medium
  - Torso length: Standard
  - Leg length: Slightly longer (anime-style)

**Clothing - Top**
- [ ] Navigate to: Outfit → Tops
- [ ] Choose style:
  - Option 1: Hoodie/casual (modern AI companion)
  - Option 2: Shirt + jacket (professional but friendly)
  - Option 3: Futuristic tech outfit
- [ ] Color: Match or complement hair color
- [ ] Add patterns/details (optional)

**Clothing - Bottom**
- [ ] Navigate to: Outfit → Bottoms
- [ ] Choose style to match top:
  - Pants, skirt, or shorts
- [ ] Color: Complementary to top
- [ ] Length: Your preference

**Accessories**
- [ ] Navigate to: Outfit → Accessories
- [ ] Add tech-themed items (optional):
  - Headphones (for AI/voice theme)
  - Headset microphone
  - Tech gadgets
  - Wristbands
- [ ] Color: Match overall theme

**Preview Full Character**
- [ ] Rotate camera 360°
- [ ] Test all expression presets
- [ ] Verify clothing doesn't clip
- [ ] Check color harmony
- [ ] Test poses (if satisfied, proceed to export)

#### Export Character (10 min)

**Pre-export Check**
- [ ] Character looks good from all angles
- [ ] No clipping issues visible
- [ ] Expressions work correctly
- [ ] Colors are harmonious

**Export Settings**
- [ ] Navigate to: Camera/Exporter tab (top right)
- [ ] Click: "Export" button
- [ ] Choose: "VRM Export"
- [ ] VRM version: **VRM 0.0** (most compatible with VSeeFace)
- [ ] Settings configuration:
  - [ ] Texture size: 2048x2048 (high quality)
  - [ ] Reduce polygon count: OFF
  - [ ] Reduce blend shapes: **OFF** (important for expressions!)
  - [ ] Include meta information: ON
- [ ] Character name: "Ani"
- [ ] Author: Your name
- [ ] Version: 1.0

**Export Process**
- [ ] Click "Export"
- [ ] Save as: `F:\Ani\character\ani_character.vrm`
- [ ] Wait for export to complete (may take 2-5 minutes)
- [ ] Check file size: Should be 10-50MB
- [ ] If > 50MB: Re-export with texture size 1024x1024

**Verification**
- [ ] File exists: `F:\Ani\character\ani_character.vrm`
- [ ] File size: 10-50MB range
- [ ] No error messages during export

**Optional: Take Screenshot**
- [ ] In VRoid Studio: Pose character nicely
- [ ] Take screenshot (Windows + Shift + S)
- [ ] Save as: `F:\Ani\character\preview.png`

**Save Project**
- [ ] File → Save Project
- [ ] Confirm saved to: `F:\Ani\character\ani_project.vroid`
- [ ] Can close VRoid Studio now

**Checkpoint**: ani_character.vrm created successfully ✓

---

### ☑️ Phase 3: VSeeFace Setup (60 min)

#### Launch & Load Character (10 min)
- [ ] Navigate to `C:\Tools\VSeeFace\`
- [ ] Double-click `VSeeFace.exe`
- [ ] Wait for VSeeFace to fully load
- [ ] Click "Open VRM" button (top left)
- [ ] Browse to: `F:\Ani\character\ani_character.vrm`
- [ ] Click "Open"
- [ ] Wait for character to load (may take 30-60 seconds)
- [ ] Character appears in viewport
- [ ] No error messages displayed

#### Camera & Visual Settings (15 min)

**Camera Position**
- [ ] Click "Camera" button
- [ ] Adjust camera:
  - [ ] Position: Center character in frame
  - [ ] Zoom: Show head and upper body (torso to top of head)
  - [ ] Angle: Slightly eye-level or slightly below
- [ ] Save camera position: "Save Camera 1"

**Background Settings**
- [ ] Click "General Settings" → "Background"
- [ ] Choose background:
  - Option 1: Transparent (for OBS green screen)
  - Option 2: Solid color (white, black, or custom)
  - Option 3: Image background
- [ ] If solid color: Choose color that contrasts with character
- [ ] Apply setting

**Quality Settings**
- [ ] Click "General Settings" → "Rendering"
- [ ] Anti-aliasing: ON (4x or 8x)
- [ ] Texture quality: High
- [ ] Shadow quality: Medium-High
- [ ] VSync: ON (prevents screen tearing)
- [ ] Frame rate limit: 60 FPS
- [ ] Apply settings

**Lighting**
- [ ] Click "General Settings" → "Lighting"
- [ ] Light intensity: 1.0-1.2 (bright enough to see details)
- [ ] Ambient light: 0.3-0.5 (soft shadows)
- [ ] Rim light: ON (optional, highlights character edges)
- [ ] Preview and adjust until character looks good

**Verify Quality**
- [ ] Character is clearly visible
- [ ] No jagged edges (anti-aliasing working)
- [ ] Smooth 60 FPS (check FPS counter if available)
- [ ] Lighting looks natural

#### Test Built-in Features (15 min)

**Test Expressions (Manual)**
- [ ] Press keyboard keys 1-9
- [ ] Each key triggers different expression
- [ ] Note which keys do what:
  - [ ] Key for Happy/Joy
  - [ ] Key for Sad
  - [ ] Key for Angry
  - [ ] Key for Surprised
  - [ ] Key for Neutral
- [ ] Verify expressions are clearly visible
- [ ] Verify smooth transitions between expressions

**Test Microphone Lip-sync**
- [ ] Click "General Settings" → "Audio"
- [ ] Enable: "Lip sync from microphone"
- [ ] Select your microphone device
- [ ] Speak into microphone: "Hello, this is a test"
- [ ] Verify character's mouth opens and closes
- [ ] Check lip-sync quality (should match speech rhythm)
- [ ] Adjust "Lip sync sensitivity" if needed (0.5-1.5)
- [ ] Speak Chinese: "你好，测试一下"
- [ ] Verify lip-sync works for Chinese too

**Test Idle Animations**
- [ ] Click "General Settings" → "Idle Animation"
- [ ] Enable: "Automatic blinking" (ON)
- [ ] Blink interval: 3-5 seconds
- [ ] Enable: "Breathing animation" (ON)
- [ ] Watch character for 30 seconds
- [ ] Verify:
  - [ ] Eyes blink naturally
  - [ ] Chest moves slightly (breathing)
  - [ ] Head sways slightly (optional)

**Optional: Test Webcam Tracking**
- [ ] Click "General Settings" → "Tracking"
- [ ] Enable: "Use webcam for tracking"
- [ ] Select your webcam
- [ ] Move your head: Character should follow
- [ ] Test expressions: Make facial expressions
- [ ] Note: We'll disable this later (using VMC instead)
- [ ] Disable webcam tracking for now

#### Enable VMC Protocol (20 min)

**VMC Receiver Setup**
- [ ] Click "General Settings" → "OSC/VMC"
- [ ] Enable: **"VMC protocol receiver"** (check the box)
- [ ] Port: **39539** (default, don't change)
- [ ] IP address: **127.0.0.1** (localhost)
- [ ] Protocol: **OSC/UDP**
- [ ] Apply settings

**Blend Shape Configuration**
- [ ] Click "General Settings" → "Blend Shapes"
- [ ] Verify all expression blend shapes are listed:
  - [ ] "Joy" or "Happy" or similar
  - [ ] "Sorrow" or "Sad" or similar
  - [ ] "Angry" or "Anger" or similar
  - [ ] "Surprised" or "Surprise" or similar
  - [ ] "Neutral" or "Relaxed" or similar
- [ ] Note exact blend shape names (write them down!):
  - Joy: _______________
  - Sad: _______________
  - Anger: _______________
  - Surprise: _______________
  - Neutral: _______________
- [ ] These names will be used in Python code

**VMC Connection Verification**
- [ ] VMC receiver is running (green indicator or "Connected")
- [ ] Port 39539 is listening
- [ ] No firewall blocking port (check Windows Firewall)
- [ ] Note connection details for Python:
  - IP: 127.0.0.1
  - Port: 39539
  - Protocol: OSC/UDP

**Save VSeeFace Configuration**
- [ ] Click "Save Settings"
- [ ] Configuration saved to VSeeFace default location
- [ ] Keep VSeeFace running for next phase

**Checkpoint**: VSeeFace ready to receive VMC commands ✓

---

## 🌤️ AFTERNOON SESSION (1:00 PM - 4:00 PM)

### ☑️ Phase 4: Python Integration (90 min)

#### Prepare Development Environment (10 min)
- [ ] Open VS Code (or your IDE)
- [ ] Open folder: `F:\Ani\`
- [ ] Verify VSeeFace is still running in background
- [ ] Open terminal in F:\Ani\

#### Create Animation Controller File (40 min)

**File Creation**
- [ ] Create new file: `animation_controller.py`
- [ ] File location: `F:\Ani\animation_controller.py`

**Implement AnimationController Class**
- [ ] Import required libraries (python-osc, asyncio, logging)
- [ ] Create AnimationController class
- [ ] Initialize VMC OSC client
- [ ] Implement methods:
  - [ ] `__init__()` - Initialize connection
  - [ ] `connect()` - Connect to VSeeFace
  - [ ] `set_expression()` - Send expression command
  - [ ] `reset_expression()` - Reset to neutral
  - [ ] `trigger_lipsync()` - Trigger lip-sync (if needed)
  - [ ] `_send_blend_shape()` - Internal OSC message sender
  - [ ] `close()` - Cleanup connection

**Expression Mapping**
- [ ] Map emotion strings to blend shape names:
  - "joy" → "Joy" (or your noted blend shape name)
  - "sad" → "Sad" (or your noted blend shape name)
  - "anger" → "Angry" (or your noted blend shape name)
  - "surprise" → "Surprised" (or your noted blend shape name)
  - "neutral" → "Neutral" (or your noted blend shape name)
- [ ] Implement intensity mapping (0.0-1.0 → blend value)

**Error Handling**
- [ ] Try-except blocks for OSC connection
- [ ] Graceful degradation if VSeeFace not running
- [ ] Logging for debugging
- [ ] Reconnection logic

**Test Animation Controller Standalone**
- [ ] Create test script in file:
  ```python
  if __name__ == "__main__":
      # Test code here
  ```
- [ ] Test connecting to VSeeFace
- [ ] Test sending each emotion
- [ ] Test intensity variations
- [ ] Verify expressions change in VSeeFace
- [ ] Fix any bugs

**Checkpoint**: animation_controller.py working standalone ✓

#### Update Main Backend Integration (40 min)

**Backup Current Code**
- [ ] Copy `main_full.py` to `main_full_backup.py`
- [ ] Safety backup created

**Import Animation Controller**
- [ ] Add to imports section:
  ```python
  from animation_controller import AnimationController
  ```

**Initialize Animation Controller**
- [ ] Find app startup section
- [ ] Add global variable: `animation_controller = None`
- [ ] Add initialization code:
  ```python
  try:
      animation_controller = AnimationController()
      logger.info("Animation controller initialized")
  except Exception as e:
      logger.warning(f"Animation controller failed: {e}")
      animation_controller = None
  ```

**Integrate with LLM Response Handler**
- [ ] Find where LLM generates response
- [ ] Find where emotion is extracted
- [ ] Add animation trigger after emotion detection:
  ```python
  if animation_controller and emotion:
      await animation_controller.set_expression(emotion, intensity)
  ```

**Integrate with TTS Audio Playback**
- [ ] Find where TTS audio is generated
- [ ] Audio already plays via browser
- [ ] VSeeFace will detect system audio for lip-sync automatically
- [ ] No additional code needed (or minimal trigger)

**Add Cleanup Handler**
- [ ] Find app shutdown section
- [ ] Add cleanup:
  ```python
  if animation_controller:
      animation_controller.close()
  ```

**Add Graceful Degradation**
- [ ] Wrap all animation calls in try-except
- [ ] Log errors but don't crash if animation fails
- [ ] System works without VSeeFace (voice-only mode)

**Test Modified Backend**
- [ ] Save all changes
- [ ] Don't start server yet (will test next phase)

**Checkpoint**: main_full.py updated with animation integration ✓

---

### ☑️ Phase 5: Integration Testing (90 min)

#### Pre-Test Setup (10 min)
- [ ] VSeeFace is running with character loaded
- [ ] VMC receiver enabled (port 39539)
- [ ] Terminal open in F:\Ani\
- [ ] Browser ready to open http://localhost:8000

#### Start Full System (10 min)
- [ ] In terminal: `python main_full.py`
- [ ] Wait for server to start
- [ ] Check logs for:
  - [ ] "Animation controller initialized" (or warning if failed)
  - [ ] Server running on http://localhost:8000
  - [ ] No critical errors
- [ ] Open browser: http://localhost:8000
- [ ] Verify web UI loads correctly

#### Basic Functionality Test (20 min)

**Test 1: English Conversation with Joy**
- [ ] Click "Start Listening" in browser
- [ ] Say: "Hello Ani, tell me something happy"
- [ ] Observe:
  - [ ] Status: "Listening..." (blue)
  - [ ] Status: "Thinking..." (purple)
  - [ ] LLM generates response
  - [ ] Emotion detected: joy
  - [ ] **VSeeFace character shows happy expression**
  - [ ] TTS plays audio
  - [ ] **Character lip-syncs to audio**
  - [ ] Returns to neutral expression after
- [ ] Record result: PASS / FAIL
- [ ] If FAIL: Note issue

**Test 2: Chinese Conversation with Sadness**
- [ ] Click "Start Listening"
- [ ] Say: "给我讲一个悲伤的故事" (Tell me a sad story)
- [ ] Observe:
  - [ ] LLM responds in Chinese
  - [ ] Emotion detected: sad
  - [ ] **Character shows sad expression**
  - [ ] Lip-sync works for Chinese audio
- [ ] Record result: PASS / FAIL

**Test 3: Surprise Emotion**
- [ ] Click "Start Listening"
- [ ] Say: "Tell me something surprising"
- [ ] Observe:
  - [ ] Emotion detected: surprise
  - [ ] **Character shows surprised expression**
- [ ] Record result: PASS / FAIL

**Test 4: Anger Emotion**
- [ ] Click "Start Listening"
- [ ] Say: "What makes you angry?"
- [ ] Observe:
  - [ ] Emotion detected: anger
  - [ ] **Character shows angry expression**
- [ ] Record result: PASS / FAIL

**Test 5: Neutral/No Strong Emotion**
- [ ] Click "Start Listening"
- [ ] Say: "What is 2 plus 2?"
- [ ] Observe:
  - [ ] Emotion detected: neutral (or low intensity)
  - [ ] Character remains neutral or subtle expression
- [ ] Record result: PASS / FAIL

**Debug Issues if Any Tests Failed**
- [ ] Check console logs in terminal
- [ ] Check browser console (F12)
- [ ] Check VSeeFace status (VMC receiving messages?)
- [ ] Verify blend shape names match
- [ ] Adjust intensity values if expressions too subtle
- [ ] Fix bugs and re-test

#### Multi-turn Conversation Test (20 min)

**Test 6: Emotion Transitions**
- [ ] Have 5-turn conversation with varying emotions:
  1. "Tell me good news" (joy)
  2. "Now tell me bad news" (sad)
  3. "That's frustrating!" (anger)
  4. "Wait, what?!" (surprise)
  5. "Okay, thanks" (neutral)
- [ ] Observe:
  - [ ] All emotions trigger correctly
  - [ ] Smooth transitions between expressions
  - [ ] No lag or freezing
  - [ ] Lip-sync works for all responses
- [ ] Record result: PASS / FAIL

**Test 7: Long Response (30+ seconds)**
- [ ] Say: "Tell me a detailed story about artificial intelligence"
- [ ] Observe:
  - [ ] Long TTS audio plays completely
  - [ ] Lip-sync continues throughout
  - [ ] Expression held or transitions naturally
  - [ ] No audio cutoff
- [ ] Record result: PASS / FAIL

**Test 8: Rapid Conversation**
- [ ] Quick back-and-forth (don't wait for full response):
  - User: "Hi"
  - User: (interrupt) "Wait"
  - User: "Never mind"
- [ ] Observe:
  - [ ] System handles interruptions gracefully
  - [ ] Expressions update accordingly
  - [ ] No crashes
- [ ] Record result: PASS / FAIL

#### Performance Monitoring (20 min)

**System Resource Check**
- [ ] Open Task Manager (Ctrl+Shift+Esc)
- [ ] Monitor during conversation:
  - [ ] GPU usage: ___% (should be < 80%)
  - [ ] CPU usage: ___% (should be < 60%)
  - [ ] RAM usage: ___GB (note value)
  - [ ] VSeeFace FPS: ___ (should be 60fps)
- [ ] Record values

**Response Time Measurement**
- [ ] Test 3 conversations
- [ ] Measure time from end of speaking to start of audio:
  - Test 1: ___ seconds
  - Test 2: ___ seconds
  - Test 3: ___ seconds
- [ ] Average: ___ seconds (target: 5-9 seconds)
- [ ] Expression change delay: Estimate ___ ms (target: < 200ms)
- [ ] Lip-sync delay: Estimate ___ ms (target: < 100ms)

**Stability Test**
- [ ] Leave system running for 10 minutes
- [ ] Have 5-6 conversations during this time
- [ ] Monitor for:
  - [ ] No crashes
  - [ ] No memory leaks (RAM stable)
  - [ ] No performance degradation
  - [ ] Expressions still working
  - [ ] Lip-sync still working
- [ ] Record result: PASS / FAIL

#### Optimization (if needed) (20 min)

**If Expression Changes Too Slow**
- [ ] Increase OSC send rate in animation_controller.py
- [ ] Reduce expression transition time
- [ ] Use async for non-blocking updates
- [ ] Re-test

**If Lip-sync Delayed**
- [ ] Adjust VSeeFace audio delay settings
- [ ] Reduce audio buffer size
- [ ] Check system audio routing
- [ ] Re-test

**If GPU Usage Too High**
- [ ] Reduce VSeeFace rendering quality
- [ ] Lower frame rate to 30fps
- [ ] Reduce texture quality
- [ ] Re-test

**If Expressions Too Subtle**
- [ ] Increase blend shape intensity multiplier
- [ ] Test: Change intensity from 0.5 → 1.0
- [ ] Adjust in animation_controller.py
- [ ] Re-test

**Checkpoint**: All basic tests passing, performance acceptable ✓

---

## 🌆 EVENING SESSION (6:00 PM - 8:00 PM)

### ☑️ Phase 6: OBS Studio Setup (60 min)

#### Launch OBS & Initial Setup (10 min)
- [ ] Launch OBS Studio
- [ ] Skip auto-configuration wizard (if appears)
- [ ] Close any existing scenes
- [ ] Ready to create new scene

#### Create Scene Collection (10 min)

**Scene Collection**
- [ ] Click: "Scene Collection" → "New"
- [ ] Name: "Ani Display"
- [ ] Click "OK"
- [ ] New collection created

**Create Main Scene**
- [ ] In Scenes panel (bottom left)
- [ ] Click "+" to add scene
- [ ] Name: "Ani Character"
- [ ] Click "OK"
- [ ] Scene created and selected

**Set Canvas Resolution**
- [ ] Settings → Video
- [ ] Base (Canvas) Resolution:
  - For 1080p TV: **1920x1080**
  - For 4K TV: **3840x2160**
  - Match your TV's native resolution
- [ ] Output (Scaled) Resolution: Same as base
- [ ] FPS: 60 (or 30 if performance issues)
- [ ] Apply settings

#### Add Sources to Scene (25 min)

**Add Window Capture for VSeeFace**
- [ ] In Sources panel (bottom middle)
- [ ] Click "+" → "Window Capture"
- [ ] Name: "VSeeFace Character"
- [ ] Click "OK"
- [ ] Properties window opens:
  - [ ] Window: Select "VSeeFace.exe" window
  - [ ] Capture method: "Windows 10/11 (1903+)"
  - [ ] Multi-adapter Compatibility: OFF (unless issues)
  - [ ] Capture cursor: OFF
- [ ] Click "OK"
- [ ] Character should appear in OBS preview

**Position & Scale Character**
- [ ] Select "VSeeFace Character" source
- [ ] Red border appears
- [ ] Drag corners to resize:
  - [ ] Fill most of screen
  - [ ] Keep aspect ratio (hold Shift while dragging)
  - [ ] Character centered or positioned nicely
- [ ] Right-click → Transform → Fit to Screen (if needed)

**Add Background (Optional)**
- [ ] Click "+" in Sources
- [ ] Choose:
  - Image: "Image"
  - Video: "Media Source"
  - Solid Color: "Color Source"
- [ ] Name: "Background"
- [ ] Click "OK"
- [ ] Configure:
  - If Image: Browse to image file
  - If Video: Browse to video file, check "Loop"
  - If Color: Choose color (black, gradient, etc.)
- [ ] Click "OK"
- [ ] Move "Background" source below "VSeeFace Character" (drag in Sources list)
- [ ] Background appears behind character

**Add Text Overlay (Optional)**
- [ ] Click "+" → "Text (GDI+)"
- [ ] Name: "Ani Title"
- [ ] Click "OK"
- [ ] Properties:
  - [ ] Text: "Ani - AI Voice Companion" (or just "Ani")
  - [ ] Font: Modern, readable (Arial, Segoe UI)
  - [ ] Size: 48-72pt
  - [ ] Color: White or match character theme
  - [ ] Outline: ON, black, width 2-4
- [ ] Click "OK"
- [ ] Position text at top or bottom of screen

**Add Audio Visualizer (Optional)**
- [ ] Click "+" → "Audio Input Capture"
- [ ] Name: "System Audio"
- [ ] Select: Desktop Audio (system audio output)
- [ ] Click "OK"
- [ ] Or skip if audio already captured

#### Apply Filters & Effects (10 min)

**On VSeeFace Character Source**
- [ ] Right-click "VSeeFace Character" → "Filters"
- [ ] Click "+" in Effects Filters

**Add Chroma Key (if using green/solid background in VSeeFace)**
- [ ] Add: "Chroma Key"
- [ ] Key Color Type: Select background color
- [ ] Adjust Similarity/Smoothness until background removed
- [ ] Preview result
- [ ] Apply if looks good

**Add Color Correction**
- [ ] Add: "Color Correction"
- [ ] Adjust:
  - [ ] Brightness: +5 to +10 (if too dark)
  - [ ] Contrast: +5 to +10 (for pop)
  - [ ] Saturation: Neutral or +5 (enhance colors)
- [ ] Preview and adjust to taste
- [ ] Apply

**Add Sharpen Filter (Optional)**
- [ ] Add: "Sharpen"
- [ ] Sharpness: 0.10-0.30 (subtle)
- [ ] Don't over-sharpen (looks unnatural)
- [ ] Preview and adjust
- [ ] Apply

**Close Filters Window**
- [ ] Click "Close"
- [ ] Filters applied

#### Configure Output to TV (5 min)

**Audio Settings**
- [ ] Settings → Audio
- [ ] Desktop Audio Device: Select system audio output (your TV if via HDMI)
- [ ] Sample Rate: 48kHz
- [ ] Channels: Stereo
- [ ] Apply

**Test Fullscreen Projector**
- [ ] Right-click on "Ani Character" scene (in preview or scenes list)
- [ ] Select: "Fullscreen Projector (Scene)"
- [ ] Choose: Your TV display (Display 2, Display 3, etc.)
- [ ] Scene should fullscreen on TV
- [ ] Press ESC to exit fullscreen (test later with real content)

**Save Scene Collection**
- [ ] File → Save
- [ ] OBS scene saved automatically

**Checkpoint**: OBS configured and can display on TV ✓

---

### ☑️ Phase 7: Final Testing & Polish (60 min)

#### Full Pipeline Test (25 min)

**Setup Complete System**
- [ ] VSeeFace running with character loaded
- [ ] Python backend running (main_full.py)
- [ ] OBS Studio open with scene configured
- [ ] Browser open to http://localhost:8000
- [ ] OBS fullscreen to TV (right-click scene → Fullscreen Projector)

**End-to-End Test 1: English Joy**
- [ ] Stand in front of TV (simulate real usage)
- [ ] In browser: Click "Start Listening"
- [ ] Say: "Hello Ani, how are you today?"
- [ ] Watch TV and observe:
  - [ ] Character appears clearly on TV
  - [ ] Expression changes to appropriate emotion
  - [ ] Lip-sync matches audio output
  - [ ] Audio plays through TV speakers clearly
  - [ ] Character returns to idle state after
- [ ] Measure total response time: ___ seconds
- [ ] Record result: PASS / FAIL

**End-to-End Test 2: Chinese Mixed Emotions**
- [ ] Say: "给我讲一个有趣的故事" (Tell me an interesting story)
- [ ] Watch TV and observe:
  - [ ] LLM responds in Chinese
  - [ ] Expression changes during story (may have multiple emotions)
  - [ ] Lip-sync works for Chinese pronunciation
  - [ ] Audio quality good on TV
- [ ] Record result: PASS / FAIL

**End-to-End Test 3: Conversation Flow**
- [ ] Have 3-turn conversation:
  1. "What's the weather like?" (neutral/informational)
  2. "That's great news!" (joy)
  3. "Tell me more" (neutral)
- [ ] Watch TV and observe:
  - [ ] Smooth flow between turns
  - [ ] Expressions appropriate for each response
  - [ ] No visual glitches on TV
  - [ ] Audio synced properly
- [ ] Record result: PASS / FAIL

**Measure Performance Metrics**
- [ ] Expression change delay: Observe on TV, estimate ___ ms
- [ ] Lip-sync delay: Observe on TV, estimate ___ ms
- [ ] Overall visual quality on TV: Rate 1-10 ___/10
- [ ] Audio quality on TV: Rate 1-10 ___/10
- [ ] VSeeFace FPS (if visible): ___ fps

#### Polish & Refinement (25 min)

**Adjust Expression Intensity (if needed)**
- [ ] If expressions too subtle on TV:
  - [ ] Edit animation_controller.py
  - [ ] Increase intensity multiplier (e.g., 0.8 → 1.0)
  - [ ] Restart Python backend
  - [ ] Re-test
- [ ] If expressions too exaggerated:
  - [ ] Decrease intensity multiplier
  - [ ] Re-test
- [ ] Optimal intensity set

**Fine-tune Lip-sync Timing (if needed)**
- [ ] If lip-sync delayed:
  - [ ] In VSeeFace: Settings → Audio
  - [ ] Adjust "Audio delay" offset (-100ms to +100ms)
  - [ ] Test with: "This is a test sentence"
  - [ ] Adjust until perfect sync
- [ ] Optimal delay set: ___ ms

**Optimize Visual Appearance**
- [ ] In VSeeFace:
  - [ ] Adjust lighting if character too dark/bright
  - [ ] Adjust camera angle if needed
  - [ ] Check character looks good on large TV screen
- [ ] In OBS:
  - [ ] Adjust color correction filter
  - [ ] Adjust brightness/contrast for TV
  - [ ] Fine-tune character position/scale
- [ ] Optimal visual settings achieved

**Audio Balancing**
- [ ] Test TTS volume on TV speakers
- [ ] If too quiet:
  - [ ] In OBS: Audio Mixer → Increase Desktop Audio level
  - [ ] Or increase system volume
- [ ] If too loud:
  - [ ] Decrease levels accordingly
- [ ] Optimal volume: Clear but not too loud

**Test Edge Cases**
- [ ] Test with background noise (fan, music)
- [ ] Test with different speaking volumes
- [ ] Test interrupting mid-response
- [ ] Test very short responses (1 word)
- [ ] Test very long responses (30+ seconds)
- [ ] System handles all cases gracefully

#### Documentation & Finalization (10 min)

**Take Screenshots/Video**
- [ ] Screenshot of VSeeFace with character
- [ ] Screenshot of OBS scene
- [ ] Screenshot of browser UI during conversation
- [ ] Short video of working system (phone camera)
- [ ] Save to: `F:\Ani\screenshots\`

**Update README.md**
- [ ] Open F:\Ani\README.md
- [ ] Add section: "3D Character Animation"
- [ ] Document:
  - VSeeFace integration
  - Expression mapping
  - Setup instructions
  - Troubleshooting tips
- [ ] Save file

**Create Setup Guide**
- [ ] Create file: `F:\Ani\SETUP_VSEFACE.md`
- [ ] Document:
  - Software installation steps
  - VSeeFace configuration
  - VMC protocol settings
  - OBS scene setup
- [ ] Save file

**Export OBS Scene**
- [ ] In OBS: Scene Collection → Export
- [ ] Save to: `F:\Ani\config\obs_ani_scene.json`
- [ ] Can import later if needed

**Git Commit**
- [ ] In terminal (F:\Ani\):
- [ ] Check status: `git status`
- [ ] Add new files:
  ```bash
  git add animation_controller.py
  git add main_full.py
  git add DAY2_ROADMAP.md
  git add CHECKLIST.md
  git add TEST_CASES.md
  git add SETUP_VSEFACE.md
  git add README.md
  ```
- [ ] Commit with message:
  ```bash
  git commit -m "Add 3D character animation with VSeeFace integration

Features:
- VMC protocol integration for real-time expression control
- Emotion-driven facial expressions (joy, sad, anger, surprise, neutral)
- Automatic lip-sync for Chinese and English TTS
- OBS Studio scene for TV display
- Comprehensive setup guide and documentation

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
  ```
- [ ] Push to GitHub: `git push origin main`
- [ ] Verify on GitHub: https://github.com/LookJohnny/AIWaifu

**Checkpoint**: All deliverables complete and committed ✓

---

## 🎯 Final Acceptance Checklist

### ✅ Functional Completeness
- [ ] 3D character created in VRoid Studio
- [ ] Character exports as VRM file successfully
- [ ] VSeeFace loads and displays character
- [ ] VMC protocol connection established
- [ ] AnimationController.py implemented and working
- [ ] Integration with main_full.py successful
- [ ] All 5 emotions trigger correct expressions
- [ ] Lip-sync works for English
- [ ] Lip-sync works for Chinese
- [ ] OBS Studio captures VSeeFace
- [ ] OBS fullscreen displays on TV via HDMI
- [ ] Audio plays through TV speakers
- [ ] Full conversation pipeline works end-to-end

### ✅ Quality Standards
- [ ] Expression changes are clearly visible
- [ ] Expression change delay < 200ms
- [ ] Lip-sync delay < 100ms
- [ ] Character looks appealing and professional
- [ ] Animations are smooth (60fps in VSeeFace)
- [ ] No visual glitches or clipping
- [ ] Audio quality is clear on TV
- [ ] Color/lighting looks good on large screen
- [ ] Idle animations (blinking, breathing) work naturally

### ✅ Performance Metrics
- [ ] GPU usage < 80% average
- [ ] CPU usage < 60% average
- [ ] Total response time still 5-9 seconds (unchanged)
- [ ] System stable for 15+ minute sessions
- [ ] No memory leaks detected
- [ ] No crashes during extended use

### ✅ Documentation
- [ ] README.md updated with Day 2 features
- [ ] SETUP_VSEFACE.md created with instructions
- [ ] DAY2_ROADMAP.md documented
- [ ] CHECKLIST.md (this file) completed
- [ ] TEST_CASES.md with all tests
- [ ] Code comments clear and helpful
- [ ] Troubleshooting tips documented

### ✅ Code Quality
- [ ] animation_controller.py clean and well-structured
- [ ] main_full.py integration is modular
- [ ] Error handling implemented
- [ ] Graceful degradation if VSeeFace unavailable
- [ ] Logging for debugging
- [ ] No hardcoded values (use config where appropriate)
- [ ] Code follows existing project style

### ✅ Repository
- [ ] All new files added to git
- [ ] Committed with descriptive message
- [ ] Pushed to GitHub successfully
- [ ] Repository includes:
  - [ ] Source code (animation_controller.py, updated main_full.py)
  - [ ] Documentation (all .md files)
  - [ ] Configuration (VSeeFace settings notes, OBS scene export)
  - [ ] Character assets (ani_character.vrm, preview.png)

---

## 🎉 Completion Celebration

### Day 2 Achievements Unlocked! 🏆

If all checkboxes above are checked, you have successfully:

✨ **Created a custom 3D anime character** from scratch
✨ **Integrated real-time expression control** based on AI emotions
✨ **Achieved smooth lip-sync** for bilingual TTS output
✨ **Set up professional TV display** with OBS Studio
✨ **Built a complete multimodal AI companion** with voice, emotion, and visual presence

**Ani is now a fully-realized AI companion!** 🎭🗣️💜

---

## 📊 Time Tracking

| Phase | Estimated | Actual | Notes |
|-------|-----------|--------|-------|
| Software Installation | 30 min | ___ min | |
| Character Creation | 90 min | ___ min | |
| VSeeFace Setup | 60 min | ___ min | |
| Python Integration | 90 min | ___ min | |
| Integration Testing | 90 min | ___ min | |
| OBS Setup | 60 min | ___ min | |
| Final Polish | 60 min | ___ min | |
| **Total** | **480 min (8 hrs)** | **___ min** | |

**Actual completion time**: ___:___ AM/PM

---

## 🚀 What's Next? (Day 3 Preview)

Now that Day 2 is complete, potential next steps:

- [ ] **Advanced Expressions**: Multiple emotion blending
- [ ] **Gesture System**: Hand and body movements
- [ ] **Improved Lip-sync**: Phoneme-based for better accuracy
- [ ] **Web-based Character**: Three.js integration in browser
- [ ] **Mobile Support**: Control via phone app
- [ ] **Voice Activity**: Idle chatter when not in conversation
- [ ] **Personality Customization**: Different character modes
- [ ] **Multi-character System**: Multiple AI personalities

**But for now: Enjoy your amazing AI companion! You earned it!** 🎉

---

**Last Updated**: 2025-10-02
**Status**: Ready for Execution
**Completion**: ___% (update as you go)
