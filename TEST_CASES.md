# 🧪 Test Cases: Day 2 - 3D Character Animation Integration

**Project**: Ani - AI Voice Companion
**Date**: 2025-10-02
**Test Phase**: 3D Animation Integration with VSeeFace

---

## 📋 Test Overview

### Test Environment
- **OS**: Windows 11
- **GPU**: RTX 4060 8GB
- **CPU**: (Your CPU model)
- **RAM**: (Your RAM amount)
- **Python**: 3.11+
- **Browser**: Chrome/Edge (latest)

### Software Under Test
- **VRoid Studio**: Latest
- **VSeeFace**: v1.13.38+
- **OBS Studio**: v30.0+
- **Backend**: main_full.py + animation_controller.py
- **TTS**: XTTS-v2
- **LLM**: qwen2.5:7b via Ollama

### Test Data
- **Voice samples**: English and Chinese speech
- **Character**: ani_character.vrm
- **Emotions**: joy, sad, anger, surprise, neutral

---

## 🎯 Test Categories

1. **Unit Tests** - Individual component testing
2. **Integration Tests** - Component interaction testing
3. **System Tests** - End-to-end pipeline testing
4. **Performance Tests** - Speed and resource usage
5. **Edge Case Tests** - Unusual scenarios and error handling
6. **User Acceptance Tests** - Real-world usage scenarios

---

## 1️⃣ UNIT TESTS

### UT-01: VRoid Character Export
**Objective**: Verify character exports correctly from VRoid Studio

**Prerequisites**:
- VRoid Studio installed
- Character created and saved

**Test Steps**:
1. Open VRoid Studio
2. Load character project: `ani_project.vroid`
3. Navigate to Camera/Exporter tab
4. Select "VRM Export"
5. Configure settings:
   - VRM version: 0.0
   - Texture size: 2048x2048
   - Reduce blend shapes: OFF
6. Export to: `F:\Ani\character\ani_character.vrm`

**Expected Results**:
- ✅ Export completes without errors
- ✅ File created: `ani_character.vrm`
- ✅ File size: 10-50MB
- ✅ No warning messages about missing data

**Actual Results**: ________________

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### UT-02: VSeeFace VRM Loading
**Objective**: Verify VSeeFace can load VRM character

**Prerequisites**:
- VSeeFace installed
- ani_character.vrm exists

**Test Steps**:
1. Launch VSeeFace
2. Click "Open VRM"
3. Select `F:\Ani\character\ani_character.vrm`
4. Wait for loading

**Expected Results**:
- ✅ Character loads within 60 seconds
- ✅ Character displays correctly in viewport
- ✅ No error messages
- ✅ All body parts visible (head, body, limbs)
- ✅ Textures loaded correctly

**Actual Results**: ________________

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### UT-03: VSeeFace Manual Expression Test
**Objective**: Verify expressions work via keyboard hotkeys

**Prerequisites**:
- VSeeFace running with character loaded

**Test Steps**:
1. Press keyboard key "1"
2. Press keyboard key "2"
3. Press keyboard key "3"
4. Press keyboard key "4"
5. Press keyboard key "5"
6. Continue through keys 1-9

**Expected Results**:
- ✅ Each key triggers different expression
- ✅ Expression changes are clearly visible
- ✅ Transitions are smooth
- ✅ Character returns to neutral when key released
- ✅ Identify joy, sad, anger, surprise mappings

**Actual Results**:
- Key for Joy: ___
- Key for Sad: ___
- Key for Anger: ___
- Key for Surprise: ___
- Key for Neutral: ___

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### UT-04: VSeeFace Microphone Lip-sync
**Objective**: Verify lip-sync works with microphone input

**Prerequisites**:
- VSeeFace running with character loaded
- Microphone enabled in VSeeFace settings

**Test Steps**:
1. Enable "Lip sync from microphone" in settings
2. Speak into microphone: "Hello, this is a test"
3. Speak: "Testing one, two, three"
4. Speak: "你好，测试一下" (Chinese)

**Expected Results**:
- ✅ Character's mouth opens when speaking
- ✅ Mouth closes when silent
- ✅ Movement roughly matches speech rhythm
- ✅ Works for both English and Chinese
- ✅ Sensitivity can be adjusted

**Actual Results**: ________________

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### UT-05: VMC Protocol Connection
**Objective**: Verify VMC receiver is enabled and listening

**Prerequisites**:
- VSeeFace running with character loaded

**Test Steps**:
1. Open VSeeFace settings
2. Navigate to "OSC/VMC" section
3. Enable "VMC protocol receiver"
4. Set port: 39539
5. Set IP: 127.0.0.1
6. Apply settings

**Expected Results**:
- ✅ VMC receiver shows "enabled" or green status
- ✅ Port 39539 is listening (check with netstat)
- ✅ No firewall blocking warnings
- ✅ Settings save successfully

**Actual Results**: ________________

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### UT-06: AnimationController Initialization
**Objective**: Verify AnimationController class initializes correctly

**Prerequisites**:
- python-osc installed
- animation_controller.py created

**Test Steps**:
1. Open Python terminal
2. Run:
   ```python
   from animation_controller import AnimationController
   controller = AnimationController()
   ```

**Expected Results**:
- ✅ No import errors
- ✅ Class instantiates successfully
- ✅ OSC client initialized
- ✅ No connection errors (or graceful warning if VSeeFace not running)

**Actual Results**: ________________

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### UT-07: AnimationController Expression Commands
**Objective**: Verify AnimationController can send expression commands

**Prerequisites**:
- VSeeFace running with VMC enabled
- AnimationController initialized

**Test Steps**:
1. In Python terminal:
   ```python
   from animation_controller import AnimationController
   import asyncio

   controller = AnimationController()

   # Test each emotion
   asyncio.run(controller.set_expression("joy", 1.0))
   # Wait 2 seconds, observe VSeeFace

   asyncio.run(controller.set_expression("sad", 1.0))
   # Wait 2 seconds, observe VSeeFace

   asyncio.run(controller.set_expression("anger", 1.0))
   # Wait 2 seconds, observe VSeeFace

   asyncio.run(controller.set_expression("surprise", 1.0))
   # Wait 2 seconds, observe VSeeFace

   asyncio.run(controller.set_expression("neutral", 1.0))
   # Wait 2 seconds, observe VSeeFace
   ```

**Expected Results**:
- ✅ "joy" → Character shows happy expression
- ✅ "sad" → Character shows sad expression
- ✅ "anger" → Character shows angry expression
- ✅ "surprise" → Character shows surprised expression
- ✅ "neutral" → Character returns to neutral
- ✅ Each command executes without errors

**Actual Results**: ________________

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### UT-08: AnimationController Intensity Scaling
**Objective**: Verify expression intensity can be controlled

**Prerequisites**:
- VSeeFace running with VMC enabled
- AnimationController initialized

**Test Steps**:
1. In Python terminal:
   ```python
   # Test different intensities for joy
   asyncio.run(controller.set_expression("joy", 0.2))  # 20%
   # Observe expression

   asyncio.run(controller.set_expression("joy", 0.5))  # 50%
   # Observe expression

   asyncio.run(controller.set_expression("joy", 1.0))  # 100%
   # Observe expression
   ```

**Expected Results**:
- ✅ Lower intensity (0.2) → Subtle expression
- ✅ Medium intensity (0.5) → Moderate expression
- ✅ High intensity (1.0) → Full expression
- ✅ Expression strength visibly scales with intensity

**Actual Results**: ________________

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

## 2️⃣ INTEGRATION TESTS

### IT-01: Backend Starts with Animation Controller
**Objective**: Verify main_full.py initializes AnimationController on startup

**Prerequisites**:
- VSeeFace running (or not, to test graceful degradation)
- main_full.py updated with animation integration

**Test Steps**:
1. Open terminal in F:\Ani\
2. Run: `python main_full.py`
3. Check console output

**Expected Results**:
- ✅ Server starts successfully
- ✅ Console shows "Animation controller initialized" (or warning if VSeeFace not running)
- ✅ Server runs on http://localhost:8000
- ✅ No critical errors
- ✅ If VSeeFace not running: Warning logged but server continues

**Actual Results**: ________________

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### IT-02: LLM Emotion Detection Triggers Expression
**Objective**: Verify LLM emotion output triggers AnimationController

**Prerequisites**:
- VSeeFace running with character
- Backend running
- Browser open to http://localhost:8000

**Test Steps**:
1. Click "Start Listening"
2. Say: "Tell me something that makes you happy"
3. Wait for LLM response
4. Observe console logs and VSeeFace

**Expected Results**:
- ✅ LLM generates response
- ✅ Emotion detected in console: "joy" or similar
- ✅ AnimationController called with emotion
- ✅ VSeeFace character shows happy expression
- ✅ Timing: Expression changes during or just before speech

**Actual Results**: ________________

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### IT-03: TTS Audio Triggers Lip-sync
**Objective**: Verify TTS audio output triggers lip-sync in VSeeFace

**Prerequisites**:
- VSeeFace running with character
- Backend running
- Browser open

**Test Steps**:
1. Click "Start Listening"
2. Say: "Count to five"
3. Wait for TTS response
4. Observe character mouth during audio playback

**Expected Results**:
- ✅ TTS audio plays through system
- ✅ VSeeFace detects audio (via system audio monitoring)
- ✅ Character's mouth moves in sync with audio
- ✅ Lip-sync delay < 100ms
- ✅ Mouth closes when audio ends

**Actual Results**:
- Lip-sync delay estimate: ___ ms

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### IT-04: Expression + Lip-sync Simultaneous
**Objective**: Verify expression and lip-sync work together without conflict

**Prerequisites**:
- Full system running

**Test Steps**:
1. Click "Start Listening"
2. Say: "Tell me a happy story" (triggers joy + speech)
3. Observe character during response

**Expected Results**:
- ✅ Character shows happy expression
- ✅ While showing expression, lip-sync still works
- ✅ No conflicts or glitches
- ✅ Expression held throughout speech
- ✅ Returns to neutral after speech ends

**Actual Results**: ________________

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### IT-05: OBS Captures VSeeFace Output
**Objective**: Verify OBS can capture and display VSeeFace window

**Prerequisites**:
- VSeeFace running with character
- OBS Studio running

**Test Steps**:
1. In OBS, create Window Capture source
2. Select VSeeFace window
3. Observe OBS preview

**Expected Results**:
- ✅ VSeeFace window captured
- ✅ Character visible in OBS preview
- ✅ No black screen or blank capture
- ✅ Animations play smoothly in OBS (60fps)
- ✅ Character fills frame appropriately

**Actual Results**: ________________

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### IT-06: OBS Fullscreen to TV
**Objective**: Verify OBS can fullscreen scene to TV display

**Prerequisites**:
- OBS scene configured
- TV connected via HDMI

**Test Steps**:
1. In OBS, right-click scene
2. Select "Fullscreen Projector (Scene)"
3. Choose TV display (Display 2/3/etc.)
4. Observe TV

**Expected Results**:
- ✅ Scene displays fullscreen on TV
- ✅ Character visible and clear
- ✅ No black bars or distortion
- ✅ Resolution matches TV native (1080p or 4K)
- ✅ Smooth playback (no stuttering)

**Actual Results**: ________________

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### IT-07: OBS Audio Routing to TV
**Objective**: Verify TTS audio plays through TV speakers

**Prerequisites**:
- Full system running
- OBS fullscreen on TV

**Test Steps**:
1. In browser, start conversation
2. Say: "Say something"
3. Listen to TV speakers

**Expected Results**:
- ✅ TTS audio plays through TV speakers
- ✅ Audio is clear and understandable
- ✅ Volume is appropriate (not too loud/quiet)
- ✅ No distortion or crackling
- ✅ Synchronized with character lip-sync on TV

**Actual Results**: ________________

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

## 3️⃣ SYSTEM TESTS (End-to-End)

### ST-01: Full Pipeline - English with Joy
**Objective**: Test complete conversation flow in English with positive emotion

**Prerequisites**:
- All systems running (VSeeFace, Backend, OBS on TV)
- Browser open

**Test Steps**:
1. Click "Start Listening" in browser
2. Say clearly: "Hello Ani, tell me some good news about technology"
3. Observe entire pipeline from input to output on TV

**Expected Results**:
- ✅ Speech recognized correctly
- ✅ Status: "Listening..." (blue) → "Thinking..." (purple) → "Speaking..." (green)
- ✅ LLM generates relevant response
- ✅ Emotion detected: joy (or positive emotion)
- ✅ Character on TV shows happy expression
- ✅ TTS audio plays with correct voice (wzy.wav)
- ✅ Lip-sync matches audio on TV
- ✅ Character returns to neutral idle state
- ✅ Total time: 5-9 seconds from end of speech to start of audio

**Actual Results**:
- Speech recognized: ________________
- Emotion detected: ________________
- Expression shown: ________________
- Total response time: ___ seconds
- Lip-sync quality: ☐ Excellent ☐ Good ☐ Fair ☐ Poor

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### ST-02: Full Pipeline - Chinese with Sadness
**Objective**: Test complete conversation flow in Chinese with sad emotion

**Prerequisites**:
- All systems running

**Test Steps**:
1. Click "Start Listening"
2. Say in Chinese: "给我讲一个悲伤的故事" (Tell me a sad story)
3. Observe pipeline

**Expected Results**:
- ✅ Chinese speech recognized correctly
- ✅ LLM responds in Chinese
- ✅ Emotion detected: sad
- ✅ Character on TV shows sad expression
- ✅ TTS Chinese pronunciation correct
- ✅ Lip-sync works for Chinese speech
- ✅ Full pipeline works same as English

**Actual Results**:
- Speech recognized: ________________
- Emotion detected: ________________
- Response language: ________________
- Lip-sync quality: ☐ Excellent ☐ Good ☐ Fair ☐ Poor

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### ST-03: Full Pipeline - Surprise Emotion
**Objective**: Test surprise emotion expression

**Prerequisites**:
- All systems running

**Test Steps**:
1. Click "Start Listening"
2. Say: "Tell me something really surprising about AI"
3. Observe response

**Expected Results**:
- ✅ LLM generates surprising fact
- ✅ Emotion detected: surprise
- ✅ Character shows surprised expression (wide eyes, open mouth)
- ✅ Expression is clearly different from other emotions
- ✅ Lip-sync continues to work

**Actual Results**: ________________

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### ST-04: Full Pipeline - Anger Emotion
**Objective**: Test anger emotion expression

**Prerequisites**:
- All systems running

**Test Steps**:
1. Click "Start Listening"
2. Say: "What makes AI systems frustrated or angry?"
3. Observe response

**Expected Results**:
- ✅ LLM generates relevant response
- ✅ Emotion detected: anger
- ✅ Character shows angry expression (furrowed brows, frown)
- ✅ Expression is clearly different from other emotions
- ✅ Lip-sync continues to work

**Actual Results**: ________________

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### ST-05: Full Pipeline - Neutral/Informational
**Objective**: Test neutral responses without strong emotion

**Prerequisites**:
- All systems running

**Test Steps**:
1. Click "Start Listening"
2. Say: "What is 15 multiplied by 7?"
3. Observe response

**Expected Results**:
- ✅ LLM provides factual answer
- ✅ Emotion detected: neutral (or low intensity)
- ✅ Character remains neutral or subtle expression
- ✅ Lip-sync works normally
- ✅ No inappropriate emotional expression

**Actual Results**: ________________

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### ST-06: Multi-turn Conversation
**Objective**: Test conversation flow across multiple turns with varying emotions

**Prerequisites**:
- All systems running

**Test Steps**:
1. Turn 1: "Hello, how are you?" (neutral/joy)
2. Turn 2: "Tell me something sad" (sad)
3. Turn 3: "Now tell me something happy" (joy)
4. Turn 4: "Give me a surprise" (surprise)
5. Turn 5: "What's frustrating about that?" (anger)

**Expected Results**:
- ✅ All 5 turns complete successfully
- ✅ Each emotion triggers correctly
- ✅ Expressions transition smoothly between turns
- ✅ No lag or performance degradation
- ✅ Chat history updates correctly
- ✅ Character returns to neutral between turns

**Actual Results**:
- Turn 1 emotion: ________________
- Turn 2 emotion: ________________
- Turn 3 emotion: ________________
- Turn 4 emotion: ________________
- Turn 5 emotion: ________________

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### ST-07: Long Response (30+ seconds)
**Objective**: Test system with extended TTS audio

**Prerequisites**:
- All systems running

**Test Steps**:
1. Click "Start Listening"
2. Say: "Tell me a detailed story about the future of artificial intelligence, at least 5 sentences long"
3. Observe entire response

**Expected Results**:
- ✅ LLM generates long response (30+ seconds audio)
- ✅ Emotion detected and expression shown
- ✅ Lip-sync continues throughout entire duration
- ✅ Expression held or transitions naturally
- ✅ Audio doesn't cut off prematurely
- ✅ Character doesn't freeze or glitch
- ✅ Returns to neutral when complete

**Actual Results**:
- Response duration: ___ seconds
- Lip-sync consistency: ☐ Perfect ☐ Good ☐ Degraded

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

## 4️⃣ PERFORMANCE TESTS

### PT-01: Response Time Measurement
**Objective**: Measure total response time for conversation

**Prerequisites**:
- All systems running

**Test Steps**:
1. Prepare stopwatch/timer
2. Click "Start Listening"
3. Say: "Hello"
4. Start timer when you stop speaking
5. Stop timer when TTS audio starts
6. Repeat 5 times for average

**Expected Results**:
- ✅ Average response time: 5-9 seconds
- ✅ Consistent across multiple tests
- ✅ No significant outliers

**Actual Results**:
- Test 1: ___ seconds
- Test 2: ___ seconds
- Test 3: ___ seconds
- Test 4: ___ seconds
- Test 5: ___ seconds
- **Average: ___ seconds**

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### PT-02: Expression Change Latency
**Objective**: Measure time from emotion detection to expression change

**Prerequisites**:
- All systems running

**Test Steps**:
1. Monitor console logs (show timestamps)
2. Start conversation
3. Note timestamp when "Emotion detected: joy" logged
4. Note timestamp when expression changes in VSeeFace
5. Calculate difference

**Expected Results**:
- ✅ Expression change delay: < 200ms
- ✅ Perceptually instant

**Actual Results**:
- Expression change delay: ___ ms

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### PT-03: Lip-sync Delay
**Objective**: Measure delay between audio and lip movement

**Prerequisites**:
- All systems running

**Test Steps**:
1. Play simple response: "One, two, three, four, five"
2. Watch character mouth closely
3. Estimate delay between audio start and mouth movement
4. Repeat multiple times

**Expected Results**:
- ✅ Lip-sync delay: < 100ms
- ✅ Perceptually synchronized
- ✅ No noticeable lag

**Actual Results**:
- Estimated delay: ___ ms
- Perception: ☐ Perfect sync ☐ Slight delay ☐ Noticeable lag

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### PT-04: GPU Usage
**Objective**: Monitor GPU utilization during operation

**Prerequisites**:
- All systems running
- Task Manager open (Performance → GPU)

**Test Steps**:
1. Baseline: Note GPU usage with just VSeeFace idle: ___%
2. Start conversation
3. During LLM processing: Note GPU usage: ___%
4. During TTS generation: Note GPU usage: ___%
5. During idle between turns: Note GPU usage: ___%

**Expected Results**:
- ✅ Average GPU usage: < 80%
- ✅ No sustained 100% usage
- ✅ GPU has headroom for other tasks

**Actual Results**:
- VSeeFace idle: ___%
- During LLM: ___%
- During TTS: ___%
- During conversation: ___% average
- Peak usage: ___%

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### PT-05: CPU Usage
**Objective**: Monitor CPU utilization during operation

**Prerequisites**:
- All systems running
- Task Manager open (Performance → CPU)

**Test Steps**:
1. Baseline: Note CPU usage at idle: ___%
2. During conversation: Note average CPU usage: ___%
3. Peak CPU during LLM processing: ___%

**Expected Results**:
- ✅ Average CPU usage: < 60%
- ✅ Peak CPU: < 90%
- ✅ System remains responsive

**Actual Results**:
- Idle CPU: ___%
- Average during conversation: ___%
- Peak CPU: ___%

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### PT-06: Memory Usage
**Objective**: Monitor RAM usage and detect memory leaks

**Prerequisites**:
- All systems running
- Task Manager open (Performance → Memory)

**Test Steps**:
1. Note initial RAM usage: ___ GB
2. Run 10 conversations (various lengths)
3. Note RAM usage after each conversation
4. Check for increasing trend

**Expected Results**:
- ✅ RAM usage stable or minor increase
- ✅ No memory leak (unbounded growth)
- ✅ System doesn't run out of memory

**Actual Results**:
- Initial RAM: ___ GB
- After 5 conversations: ___ GB
- After 10 conversations: ___ GB
- Trend: ☐ Stable ☐ Slight increase ☐ Concerning growth

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### PT-07: VSeeFace Frame Rate
**Objective**: Verify VSeeFace maintains 60 FPS

**Prerequisites**:
- VSeeFace running with character

**Test Steps**:
1. Check VSeeFace settings for FPS display (enable if available)
2. Or estimate visually (smooth = 60fps, choppy = lower)
3. Monitor FPS during:
   - Idle: ___ fps
   - Expression changes: ___ fps
   - Lip-sync: ___ fps

**Expected Results**:
- ✅ Maintains 60 FPS consistently
- ✅ No dropped frames during animations
- ✅ Smooth visual experience

**Actual Results**:
- Idle FPS: ___ fps
- During animation: ___ fps
- Assessment: ☐ Smooth (60fps) ☐ Acceptable (30-60fps) ☐ Choppy (<30fps)

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### PT-08: Stability Test (15 minutes)
**Objective**: Verify system runs stably for extended period

**Prerequisites**:
- All systems running

**Test Steps**:
1. Start timer for 15 minutes
2. Have 6-8 conversations during this period (varied content)
3. Monitor for:
   - Crashes
   - Error messages
   - Performance degradation
   - Memory/resource issues

**Expected Results**:
- ✅ No crashes in any component
- ✅ No error messages or warnings
- ✅ Performance remains consistent
- ✅ All features continue working
- ✅ System responsive throughout

**Actual Results**:
- Conversations completed: ___
- Crashes: ☐ None ☐ VSeeFace ☐ Backend ☐ OBS ☐ Browser
- Errors: ☐ None ☐ Minor ☐ Major
- Performance: ☐ Stable ☐ Degraded slightly ☐ Degraded significantly

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

## 5️⃣ EDGE CASE TESTS

### EC-01: VSeeFace Not Running
**Objective**: Verify graceful degradation when VSeeFace unavailable

**Prerequisites**:
- VSeeFace CLOSED (not running)
- Backend ready to start

**Test Steps**:
1. Ensure VSeeFace is not running
2. Start backend: `python main_full.py`
3. Check console output
4. Try conversation in browser

**Expected Results**:
- ✅ Backend starts successfully (doesn't crash)
- ✅ Console shows warning: "Animation controller unavailable" or similar
- ✅ Voice conversation still works (voice-only mode)
- ✅ No critical errors
- ✅ System functional without animation

**Actual Results**: ________________

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### EC-02: VMC Connection Lost Mid-Session
**Objective**: Test recovery when VSeeFace closes during operation

**Prerequisites**:
- Full system running with active conversation

**Test Steps**:
1. Start a conversation
2. During response, close VSeeFace
3. Observe backend behavior
4. Try another conversation

**Expected Results**:
- ✅ Backend logs warning about lost connection
- ✅ Backend doesn't crash
- ✅ Voice conversation continues working
- ✅ Reconnects if VSeeFace restarted

**Actual Results**: ________________

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### EC-03: Rapid Expression Changes
**Objective**: Test system with multiple emotions in quick succession

**Prerequisites**:
- All systems running

**Test Steps**:
1. Craft prompt that triggers multiple emotions: "Tell me a story that starts happy, becomes sad, then surprising, then angry, and ends neutral"
2. Observe expression changes during long response

**Expected Results**:
- ✅ Multiple expressions detected
- ✅ Character transitions smoothly between expressions
- ✅ No expression conflicts or glitches
- ✅ Lip-sync continues throughout
- ✅ No crashes

**Actual Results**:
- Number of expression changes: ___
- Transitions: ☐ Smooth ☐ Acceptable ☐ Glitchy

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### EC-04: Very Short Response (1-2 words)
**Objective**: Test with minimal TTS output

**Prerequisites**:
- All systems running

**Test Steps**:
1. Click "Start Listening"
2. Say: "Say yes"
3. Observe response

**Expected Results**:
- ✅ LLM responds with short answer
- ✅ Expression triggers (even if brief)
- ✅ Lip-sync works for short audio
- ✅ Character returns to neutral quickly
- ✅ No errors

**Actual Results**: ________________

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### EC-05: Interruption During Speech
**Objective**: Test user interrupting AI response

**Prerequisites**:
- All systems running

**Test Steps**:
1. Start conversation with long response: "Tell me a long story"
2. While AI is speaking, click "Start Listening" again (interrupt)
3. Say: "Stop, tell me something else"

**Expected Results**:
- ✅ Audio playback stops (or continues, depends on implementation)
- ✅ New speech recognized
- ✅ New response generated
- ✅ Expression updates accordingly
- ✅ No crashes or stuck states

**Actual Results**: ________________

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### EC-06: No Microphone Input (Silence)
**Objective**: Test system when user says nothing

**Prerequisites**:
- All systems running

**Test Steps**:
1. Click "Start Listening"
2. Don't say anything for 10 seconds
3. Observe behavior

**Expected Results**:
- ✅ System times out gracefully
- ✅ Returns to idle state
- ✅ No error shown to user (or helpful message)
- ✅ Character remains in neutral/idle

**Actual Results**: ________________

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### EC-07: Background Noise Interference
**Objective**: Test with ambient noise

**Prerequisites**:
- All systems running
- Noise source (music, fan, TV in background)

**Test Steps**:
1. Play background noise at moderate volume
2. Click "Start Listening"
3. Say: "Hello, can you hear me?"
4. Observe recognition quality

**Expected Results**:
- ✅ Speech still recognized (may have minor errors)
- ✅ System doesn't recognize noise as speech
- ✅ Conversation flow continues
- ✅ Acceptable degradation in noisy environment

**Actual Results**:
- Recognition accuracy: ☐ Perfect ☐ Good ☐ Degraded ☐ Failed

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### EC-08: Mixed Language in One Response
**Objective**: Test when LLM mixes English and Chinese

**Prerequisites**:
- All systems running

**Test Steps**:
1. Click "Start Listening"
2. Say: "Tell me about 人工智能 in both English and Chinese" (mix languages)
3. Observe response

**Expected Results**:
- ✅ LLM handles mixed language request
- ✅ TTS handles both languages (may sound unnatural)
- ✅ Lip-sync works for both
- ✅ Expression triggers correctly
- ✅ No crashes

**Actual Results**: ________________

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### EC-09: OBS Window Minimized/Hidden
**Objective**: Test when OBS is not visible

**Prerequisites**:
- All systems running
- OBS minimized to taskbar

**Test Steps**:
1. Minimize OBS window
2. Have conversation
3. Restore OBS window
4. Check if display still working

**Expected Results**:
- ✅ OBS continues capturing in background
- ✅ When restored, display is still working
- ✅ Fullscreen projector still active
- ✅ No visual glitches

**Actual Results**: ________________

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### EC-10: System Resource Exhaustion
**Objective**: Test behavior when system is under heavy load

**Prerequisites**:
- All systems running
- Start resource-intensive task (e.g., video encoding, game)

**Test Steps**:
1. Start heavy background task
2. Attempt conversation
3. Monitor performance

**Expected Results**:
- ✅ System still functional (may be slower)
- ✅ No crashes
- ✅ Graceful performance degradation
- ✅ Error messages if resources unavailable

**Actual Results**:
- System responsiveness: ☐ Normal ☐ Slower ☐ Very slow ☐ Unresponsive

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

## 6️⃣ USER ACCEPTANCE TESTS

### UAT-01: First-time User Experience
**Objective**: Simulate new user trying system for first time

**Prerequisites**:
- All systems set up and running

**Test Steps**:
1. Ask someone unfamiliar with system to try it
2. Give minimal instructions: "Open browser to localhost:8000 and talk to Ani"
3. Observe their interaction

**Expected Results**:
- ✅ User understands how to start conversation
- ✅ User finds it intuitive
- ✅ Character behavior is natural and engaging
- ✅ User has positive experience

**Actual Results**:
- User feedback: ________________
- Ease of use (1-10): ___
- Enjoyment (1-10): ___

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### UAT-02: TV Display Quality
**Objective**: Evaluate visual quality on large TV screen

**Prerequisites**:
- OBS fullscreen on TV

**Test Steps**:
1. View character on TV from normal viewing distance (6-10 feet)
2. Have conversation and observe
3. Rate visual quality

**Expected Results**:
- ✅ Character is clearly visible from across room
- ✅ Expressions are noticeable and engaging
- ✅ No pixelation or blurriness
- ✅ Colors are vibrant and appealing
- ✅ Professional appearance

**Actual Results**:
- Visual clarity (1-10): ___
- Expression visibility (1-10): ___
- Overall appearance (1-10): ___

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### UAT-03: Audio Quality on TV Speakers
**Objective**: Evaluate audio quality through TV

**Prerequisites**:
- Audio routing to TV speakers

**Test Steps**:
1. Have conversation with AI
2. Listen from normal viewing distance
3. Rate audio quality

**Expected Results**:
- ✅ Voice is clear and understandable
- ✅ Volume is appropriate
- ✅ No distortion or static
- ✅ Chinese and English both clear
- ✅ Professional quality

**Actual Results**:
- Clarity (1-10): ___
- Volume appropriateness (1-10): ___
- Overall audio quality (1-10): ___

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### UAT-04: Emotional Expressiveness
**Objective**: Evaluate how well emotions are conveyed

**Prerequisites**:
- All systems running

**Test Steps**:
1. Have conversations triggering all 5 emotions
2. Rate how well each emotion is expressed

**Expected Results**:
- ✅ Joy is clearly recognizable
- ✅ Sadness is clearly recognizable
- ✅ Anger is clearly recognizable
- ✅ Surprise is clearly recognizable
- ✅ Neutral is natural baseline
- ✅ Emotions enhance conversation experience

**Actual Results**:
- Joy expressiveness (1-10): ___
- Sad expressiveness (1-10): ___
- Anger expressiveness (1-10): ___
- Surprise expressiveness (1-10): ___
- Neutral naturalness (1-10): ___
- Overall emotional engagement (1-10): ___

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### UAT-05: Lip-sync Realism
**Objective**: Evaluate perceived quality of lip-sync

**Prerequisites**:
- All systems running

**Test Steps**:
1. Have several conversations
2. Watch lip movements closely
3. Rate realism

**Expected Results**:
- ✅ Lip-sync appears natural
- ✅ Matches speech rhythm
- ✅ Not distractingly off
- ✅ Enhances immersion

**Actual Results**:
- Lip-sync realism (1-10): ___
- Immersion level (1-10): ___
- Assessment: ☐ Excellent ☐ Good ☐ Acceptable ☐ Needs work

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

### UAT-06: Overall User Satisfaction
**Objective**: Gauge overall satisfaction with Day 2 features

**Prerequisites**:
- All systems running
- Multiple test conversations completed

**Test Steps**:
1. Use system for 15-20 minutes
2. Test various features
3. Rate overall experience

**Expected Results**:
- ✅ System is engaging and fun to use
- ✅ 3D character adds value over voice-only
- ✅ User would want to use it again
- ✅ Feels like complete AI companion experience

**Actual Results**:
- Overall satisfaction (1-10): ___
- Would recommend to others: ☐ Yes ☐ No
- Favorite feature: ________________
- Improvement suggestion: ________________

**Status**: ☐ PASS ☐ FAIL ☐ SKIP

**Notes**: ________________

---

## 📈 Test Summary Report

### Test Execution Summary

| Category | Total Tests | Passed | Failed | Skipped | Pass Rate |
|----------|-------------|--------|--------|---------|-----------|
| Unit Tests (UT) | 8 | ___ | ___ | ___ | __% |
| Integration Tests (IT) | 7 | ___ | ___ | ___ | __% |
| System Tests (ST) | 7 | ___ | ___ | ___ | __% |
| Performance Tests (PT) | 8 | ___ | ___ | ___ | __% |
| Edge Case Tests (EC) | 10 | ___ | ___ | ___ | __% |
| User Acceptance (UAT) | 6 | ___ | ___ | ___ | __% |
| **TOTAL** | **46** | **___** | **___** | **___** | **___%** |

### Critical Issues Found
1. ________________
2. ________________
3. ________________

### Non-Critical Issues Found
1. ________________
2. ________________
3. ________________

### Performance Metrics Summary
- **Average Response Time**: ___ seconds (Target: 5-9s)
- **Expression Change Delay**: ___ ms (Target: <200ms)
- **Lip-sync Delay**: ___ ms (Target: <100ms)
- **Average GPU Usage**: ___% (Target: <80%)
- **Average CPU Usage**: ___% (Target: <60%)
- **VSeeFace FPS**: ___ fps (Target: 60fps)

### Overall Assessment
- ☐ **PASS** - All critical tests passed, system ready for use
- ☐ **CONDITIONAL PASS** - Minor issues found, but system functional
- ☐ **FAIL** - Critical issues prevent system from working properly

### Tester Notes
________________
________________
________________

### Next Steps
- [ ] Fix critical issues (if any)
- [ ] Address non-critical issues
- [ ] Optimize performance (if needed)
- [ ] Update documentation with findings
- [ ] Proceed to polish phase

---

**Test Date**: 2025-10-02
**Tester**: ________________
**System Version**: Day 2 - 3D Animation Integration
**Test Duration**: ___ hours
**Completion Status**: ___% complete

---

## 🎯 Acceptance Criteria

To consider Day 2 complete, the following must be TRUE:

### Must Have (Critical)
- [x] At least 90% of Unit Tests pass
- [x] At least 85% of Integration Tests pass
- [x] At least 80% of System Tests pass
- [x] All 5 emotions trigger correctly
- [x] Lip-sync works for both languages
- [x] OBS displays on TV successfully
- [x] No critical crashes or errors
- [x] Performance within acceptable limits

### Should Have (Important)
- [x] At least 70% of Performance Tests pass
- [x] At least 60% of Edge Case Tests pass
- [x] User satisfaction rating > 7/10
- [x] Visual quality rating > 7/10
- [x] Audio quality rating > 7/10

### Nice to Have (Optional)
- [ ] 100% of all tests pass
- [ ] User satisfaction > 9/10
- [ ] Performance exceeds targets
- [ ] No issues found whatsoever

**Status**: ☐ Acceptance Criteria Met ☐ Not Met

---

**Good luck with testing! This comprehensive test suite will ensure your Day 2 integration is solid and production-ready!** 🧪✨
