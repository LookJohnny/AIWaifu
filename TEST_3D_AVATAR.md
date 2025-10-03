# 3D Avatar System Testing Guide

## ✅ Server Status
Server is running successfully at **http://localhost:8000**

All pipelines initialized:
- ✅ LLM: Ollama qwen2.5:7b (bilingual EN+ZH)
- ✅ TTS: Coqui XTTS-v2 with wzy.wav voice (GPU accelerated)
- ✅ Audio: Silero VAD + Faster-Whisper (CUDA)
- ✅ Animation: Connected to VMC port 39539
- ✅ 3D Frontend: avatar_3d.html with Three.js VRM renderer

## 🎯 Testing Steps

### Phase 1: VRM Model Loading (2 minutes)

1. **Open Browser**
   ```
   http://localhost:8000
   ```

2. **Expected Behavior:**
   - Loading spinner shows "Loading Ani..."
   - After 2-5 seconds, darkhair.vrm character appears
   - Character should be:
     - Centered in view
     - Well-lit with gradient background
     - Rotatable with mouse drag
     - Zoomable with mouse wheel

3. **Success Criteria:**
   - ✅ Character loads without errors (check browser console F12)
   - ✅ "Loading Ani..." disappears
   - ✅ Controls panel appears at bottom
   - ✅ Status shows "Connected" (green dot)
   - ✅ Emotion display shows "Neutral" (top right)

4. **Console Check (F12):**
   ```javascript
   // Should see:
   [OK] VRM loaded successfully
   Available expressions: happy, sad, angry, surprised, neutral
   [OK] WebSocket connected
   ```

---

### Phase 2: Expression Testing (5 minutes)

1. **Test Manual Expressions (Browser Console F12):**
   ```javascript
   // Test joy
   ani.setExpression('joy', 1.0)

   // Test sadness
   ani.setExpression('sad', 1.0)

   // Test anger
   ani.setExpression('anger', 1.0)

   // Test surprise
   ani.setExpression('surprise', 1.0)

   // Reset to neutral
   ani.setExpression('neutral', 1.0)
   ```

2. **Expected Behavior:**
   - Character's facial expression changes
   - Emotion display updates (top right)
   - Transition is smooth (not instant)
   - Each emotion should be visually distinct

3. **Test Intensity (Optional):**
   ```javascript
   // Subtle smile
   ani.setExpression('joy', 0.3)

   // Full smile
   ani.setExpression('joy', 1.0)
   ```

---

### Phase 3: Voice Interaction (10 minutes)

#### Test Case 3.1: Simple Greeting (English)

1. **Start Interaction:**
   - Click "Start Voice" button
   - Allow microphone access when prompted
   - Status changes to "Listening..."

2. **Speak:**
   ```
   "Hello Ani, how are you?"
   ```

3. **Expected Flow:**
   - Status: "Listening..." → "Processing..." → "Ready"
   - Character expression changes based on emotion (likely "joy")
   - Audio plays with character's voice response
   - Mouth moves during speech (simple lip-sync)

4. **Success Criteria:**
   - ✅ Audio is clear and matches wzy.wav voice style
   - ✅ Expression matches emotion (joy for greeting)
   - ✅ Lip-sync animation runs during audio
   - ✅ Response is contextually appropriate

#### Test Case 3.2: Emotional Response (Chinese)

1. **Speak:**
   ```
   "我今天很开心！" (I'm very happy today!)
   ```

2. **Expected:**
   - Ani responds in Chinese
   - Expression shows "joy" (高兴)
   - Voice maintains wzy.wav character

#### Test Case 3.3: Sad Emotion

1. **Speak:**
   ```
   "I'm feeling really sad today"
   ```

2. **Expected:**
   - Expression: "sad" (眉头下垂，嘴角下垂)
   - Response shows empathy
   - Voice tone may be slightly softer

#### Test Case 3.4: Surprise Trigger

1. **Speak:**
   ```
   "Wow! That's amazing!"
   ```

2. **Expected:**
   - Expression: "surprised" (眼睛睁大，嘴巴张开)
   - Response matches excitement

---

### Phase 4: WebSocket Message Flow (Advanced)

1. **Open Browser DevTools Network Tab:**
   - Filter by "WS" (WebSocket)
   - Click on `ws://localhost:8000/ws` connection
   - Go to "Messages" tab

2. **Speak a phrase and observe message sequence:**

   **Outgoing (Client → Server):**
   ```json
   {
     "type": "audio",
     "audio": "UklGRi4AAABXQVZFZm10..." // base64 audio
   }
   ```

   **Incoming (Server → Client):**
   ```json
   // Message 1: Emotion
   {
     "type": "emotion",
     "emotion": "joy",
     "intensity": 0.85
   }

   // Message 2: Audio
   {
     "type": "audio",
     "audio": "UklGRjQEAABXQVZFZm10...",
     "text": "你好！我很高兴见到你！"
   }

   // Message 3: Complete response
   {
     "status": "success",
     "validated": true,
     "data": {
       "utterance": "你好！我很高兴见到你！",
       "emote": {
         "type": "joy",
         "intensity": 0.85
       },
       "intent": "SMALL_TALK"
     },
     "llm_latency_ms": 1243,
     "total_latency_ms": 3872
   }
   ```

3. **Verify:**
   - ✅ Emotion arrives first (immediate visual feedback)
   - ✅ Audio follows quickly
   - ✅ Character expression syncs with emotion
   - ✅ Mouth animation starts with audio playback

---

### Phase 5: Performance Testing

1. **Latency Check (Server Console):**
   ```
   [LLM Latency] ~1000-2000ms
   [TTS Latency] ~2000-4000ms
   ```

2. **Frame Rate Check (Browser Console):**
   ```javascript
   // Monitor FPS
   let lastTime = performance.now();
   let frames = 0;

   setInterval(() => {
     const now = performance.now();
     const fps = frames / ((now - lastTime) / 1000);
     console.log(`FPS: ${fps.toFixed(1)}`);
     frames = 0;
     lastTime = now;
   }, 1000);

   // Increment on animation frame
   requestAnimationFrame(function count() {
     frames++;
     requestAnimationFrame(count);
   });
   ```

   **Expected:** 55-60 FPS (smooth animation)

3. **GPU Usage:**
   - Open Task Manager → Performance → GPU
   - Should see CUDA usage during TTS generation
   - WebGL usage during 3D rendering

---

## 🐛 Troubleshooting

### Issue: Character Not Loading
**Symptoms:** Loading screen stuck, console shows 404 error
**Fix:**
```bash
# Check if character file exists
ls F:\Ani\character\darkhair.vrm

# Verify server has static file mount
curl http://localhost:8000/character/darkhair.vrm
```

### Issue: No Expression Changes
**Symptoms:** Character face frozen, expression commands don't work
**Debug:**
```javascript
// Check VRM structure
console.log(ani.vrm());
console.log(ani.vrm().expressionManager);

// Try different expression names
ani.setExpression('happy', 1.0)  // Try 'happy' instead of 'joy'
```

### Issue: WebSocket Not Connecting
**Symptoms:** Status stuck on "Disconnected"
**Fix:**
```javascript
// Manual reconnect
ani.connectWebSocket()

// Check server
fetch('http://localhost:8000/health')
  .then(r => r.json())
  .then(console.log)
```

### Issue: No Audio Playback
**Symptoms:** Expression works but silent
**Check:**
1. Browser audio permissions
2. WebSocket messages include audio field (DevTools Network)
3. Audio element creation (Console → ani.currentAudio)

### Issue: Microphone Not Working
**Fix:**
1. Chrome: chrome://settings/content/microphone
2. Allow http://localhost:8000
3. Restart browser if needed

---

## 🎨 Customization Tips

### Adjust Camera Position
```javascript
// In avatar_3d.html, modify camera setup
camera.position.set(0, 1.4, 2.5);  // Closer view
camera.position.set(0, 1.2, 4);    // Further away
```

### Change Lighting
```javascript
// Brighter ambient light
const ambientLight = new THREE.AmbientLight(0xffffff, 1.2);

// Add fill light
const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
fillLight.position.set(-1, 0, -1);
scene.add(fillLight);
```

### Improve Lip-Sync
```javascript
// Adjust mouth animation speed
mouthAnimationFrame = setTimeout(animate, 80);  // Faster
mouthAnimationFrame = setTimeout(animate, 150); // Slower
```

---

## 📊 Success Metrics

### Phase 1 (VRM Loading): ✅
- [ ] Character loads in <5 seconds
- [ ] No console errors
- [ ] Smooth 60fps rendering
- [ ] Mouse controls work

### Phase 2 (Expressions): ✅
- [ ] All 5 emotions work (joy, sad, anger, surprise, neutral)
- [ ] Facial changes are visible
- [ ] UI updates correctly

### Phase 3 (Voice): ✅
- [ ] Microphone captures audio
- [ ] LLM responds appropriately
- [ ] TTS generates clear speech
- [ ] Lip-sync runs during audio
- [ ] Expressions match emotion

### Phase 4 (Integration): ✅
- [ ] WebSocket message flow correct
- [ ] Emotion arrives before audio
- [ ] No race conditions
- [ ] Error handling works

### Phase 5 (Performance): ✅
- [ ] <2s LLM latency
- [ ] <4s TTS latency
- [ ] 55+ FPS rendering
- [ ] GPU acceleration active

---

## 🚀 Next Steps

If all tests pass:

1. **OBS Integration:**
   - Add browser source: http://localhost:8000
   - Set resolution: 1920x1080
   - Enable "Shutdown source when not visible" for GPU efficiency

2. **Advanced Lip-Sync:**
   - Integrate Rhubarb Lip-Sync for phoneme-accurate mouth shapes
   - Use TTS phoneme hints from backend

3. **Enhanced Expressions:**
   - Add blend shape combinations (happy + surprised = excited)
   - Implement emotion blending (smooth transitions)

4. **Character Customization:**
   - Replace darkhair.vrm with custom VRM
   - Add clothing/accessories support
   - Implement body animations

5. **Production Deployment:**
   - Add HTTPS for microphone access on LAN
   - Optimize VRM file size (<5MB)
   - Cache Three.js libraries locally
