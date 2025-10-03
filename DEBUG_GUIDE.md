# Complete Debug Guide - Ani 3D Avatar System

## 🔍 Step-by-Step Debugging

### Step 1: Refresh Browser
**Action:** Press F5 or Ctrl+R to reload http://localhost:8000

**Expected:**
- Loading spinner appears
- After 2-5s, character loads
- Bottom panel shows "Hold to Talk" button
- Top left: Green dot + "Ready"
- Top right: "Current Emotion: Neutral"

**If character doesn't load:**
- Open DevTools (F12) → Console tab
- Look for errors
- Check Network tab for `/character/darkhair.vrm` status (should be 200 OK)

---

### Step 2: Check VRM Expressions
**Action:** Open http://localhost:8000/debug in a new tab

**This page will show:**
1. All available expressions in darkhair.vrm
2. Test results for each expression
3. All blend shape groups

**Expected Output:**
```
VRM Loaded Successfully!

Available Expressions:
• happy
• sad
• angry
• surprised
• neutral
• aa, ee, ih, oh, ou (visemes for lip-sync)
```

**If you see different expression names**, we need to update the mapping!

---

### Step 3: Manual Expression Test
**Action:** On main page (http://localhost:8000), open Console (F12) and type:

```javascript
// Test 1: Check VRM loaded
ani.vrm()  // Should return VRM object, not null

// Test 2: Try default expressions
ani.setExpression('joy', 1.0)
// Wait 2 seconds, watch character face

ani.setExpression('sad', 1.0)
// Wait 2 seconds

ani.setExpression('surprise', 1.0)
// Wait 2 seconds

ani.setExpression('neutral', 1.0)
```

**Expected:** Character's facial expression should change visibly each time

**If face doesn't change:**
- Character might not have standard VRM expressions
- Check /debug page for actual expression names
- VRM might use custom blend shapes

---

### Step 4: Test Voice Recognition
**Action:** Click and HOLD "Hold to Talk" button

**Expected:**
1. Button turns RED with pulsing animation
2. Status changes to "Listening..." (yellow dot)
3. Mic status shows "Listening..."

**Speak:** "你好" or "Hello"

**Release button**

**Expected:**
1. Console shows `[Speech] 你好` or `[Speech] Hello`
2. Status changes to "Thinking..." (yellow dot)
3. Transcript appears: "You: 你好"

**If microphone doesn't work:**
- Browser needs microphone permission
- Check browser address bar for microphone icon
- Chrome: chrome://settings/content/microphone
- Make sure http://localhost:8000 is allowed

**If speech recognition fails:**
- Web Speech API only works in Chrome/Edge
- Firefox doesn't support it yet
- Check console for error messages

---

### Step 5: Test AI Response
**Action:** After speaking and releasing button

**Expected Flow:**
1. Status: "Thinking..." (yellow)
2. Console: `[OK] Text sent to AI: 你好`
3. Console: `[WebSocket] {type: 'emotion', emotion: 'joy', intensity: 0.8}`
4. **Character face changes to happy**
5. Console: `[WebSocket] {type: 'audio', audio: 'UklGR...'}`
6. **Audio plays** (Ani's voice)
7. **Mouth moves** during speech
8. Transcript: "Ani: 你好！很高兴见到你！"
9. Status: "Ready" (green)

**Check server console for:**
```
[User] 你好
[Ani] 你好！很高兴见到你！
[Emote] joy (0.85)
[LLM Latency] 1234ms
[TTS Latency] 2876ms
```

---

### Step 6: WebSocket Message Inspection
**Action:** Open DevTools (F12) → Network tab → Filter "WS"

**Click on:** ws://localhost:8000/ws → Messages tab

**Speak a phrase and watch messages:**

**Outgoing (you → server):**
```json
{
  "type": "user_input",
  "text": "你好"
}
```

**Incoming (server → you):**
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
  "audio": "UklGRjQEAABXQVZF...",
  "text": "你好！很高兴见到你！"
}

// Message 3: Complete response
{
  "status": "success",
  "validated": true,
  "data": {
    "utterance": "你好！很高兴见到你！",
    "emote": {
      "type": "joy",
      "intensity": 0.85
    },
    "intent": "SMALL_TALK"
  },
  "llm_latency_ms": 1234,
  "total_latency_ms": 4110
}
```

---

## 🐛 Common Issues & Solutions

### Issue 1: Character Not Moving

**Symptom:** Character loads but face never changes

**Possible Causes:**

**A) VRM doesn't have standard expressions**
- Solution: Check /debug page
- Update EMOTION_MAP in avatar_3d.html to match actual expression names

**B) Expressions exist but not updating**
- Check console for errors when calling `ani.setExpression()`
- Try directly: `ani.vrm().expressionManager.setValue('happy', 1.0)`

**C) Animation loop not running**
- Check console: `requestAnimationFrame` errors
- Try: Reload page, check GPU acceleration enabled

**Fix for custom expression names:**
```javascript
// In avatar_3d.html, update EMOTION_MAP around line 222
const EMOTION_MAP = {
    'joy': 'Joy',      // Capital J if VRM uses "Joy"
    'sad': 'Sorrow',   // Some VRMs use "Sorrow" not "sad"
    'anger': 'Angry',
    'surprise': 'Surprised',
    'neutral': 'Neutral'
};
```

---

### Issue 2: No Voice/Audio

**Symptom:** Expression changes but no sound plays

**Check:**
1. Browser volume (not muted)
2. Console errors: `[ERROR] Failed to play audio`
3. WebSocket receives audio message (DevTools → Network → WS)

**Debug:**
```javascript
// Check if audio data is received
// In Console, after speaking:
// Look for: [WebSocket] {type: 'audio', audio: '...'}
```

**If audio message not received:**
- Check server console for TTS errors
- Verify TTS pipeline initialized: `curl http://localhost:8000/health`

**If audio received but not playing:**
- Browser autoplay policy (user must interact with page first)
- Base64 decode error (console will show error)

---

### Issue 3: Speech Recognition Not Working

**Symptom:** Button click does nothing, or "Speech recognition not supported"

**Check:**
1. Browser: Must be Chrome, Edge, or Safari (not Firefox)
2. Console shows: `[ERROR] Speech Recognition not supported`

**Solution:**
- Use Chrome/Edge browser
- Or test with text input:
  ```javascript
  ani.sendText('你好')
  ```

**If microphone permission denied:**
- Chrome: chrome://settings/content/microphone
- Allow for http://localhost:8000
- Reload page

---

### Issue 4: LLM Not Responding

**Symptom:** Speech recognized but no AI response

**Check server console for:**
```
[User] 你好
// Should see [Ani] response here
```

**If nothing appears:**
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Expected: List of models including qwen2.5:7b
```

**Fix:**
```bash
# Start Ollama if not running
ollama serve

# In another terminal, test model
ollama run qwen2.5:7b "你好"
```

---

### Issue 5: Slow Response

**Symptom:** Works but takes >10 seconds

**Check latencies in server console:**
```
[LLM Latency] 5432ms  ← Too slow (should be <2000ms)
[TTS Latency] 8765ms  ← Too slow (should be <4000ms)
```

**Optimization:**

**For LLM:**
- Use smaller model: `qwen2.5:3b` instead of `7b`
- Or switch to OpenAI API (much faster)

**For TTS:**
- Switch to Edge TTS (instant but robotic):
  ```python
  # In main_full.py, line 122
  tts_config = TTSConfig(engine="edge", voice="zh-CN-XiaoyiNeural")
  ```

---

## 🔧 Advanced Debugging

### Test Individual Components

**1. Test LLM directly:**
```bash
curl -X POST http://localhost:8000/ws \
  -H "Content-Type: application/json" \
  -d '{"type": "user_input", "text": "你好"}'
```

**2. Test TTS directly:**
```bash
curl -X POST http://localhost:8000/api/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "你好"}' \
  --output test.wav

# Play test.wav to verify voice works
```

**3. Test expressions programmatically:**
```javascript
// Cycle through all emotions automatically
const emotions = ['joy', 'sad', 'anger', 'surprise', 'neutral'];
let i = 0;

setInterval(() => {
  ani.setExpression(emotions[i], 1.0);
  console.log('Testing:', emotions[i]);
  i = (i + 1) % emotions.length;
}, 2000);
```

---

## 📊 Performance Monitoring

### Check Frame Rate:
```javascript
let lastTime = performance.now();
let frames = 0;

function countFPS() {
  frames++;
  requestAnimationFrame(countFPS);
}
countFPS();

setInterval(() => {
  const now = performance.now();
  const fps = frames / ((now - lastTime) / 1000);
  console.log(`FPS: ${fps.toFixed(1)}`);
  frames = 0;
  lastTime = now;
}, 1000);
```

**Expected:** 55-60 FPS
**If <30 FPS:** GPU acceleration issue or VRM too complex

---

## 🎯 Success Checklist

Run through this checklist:

- [ ] VRM loads at http://localhost:8000
- [ ] http://localhost:8000/debug shows expression list
- [ ] `ani.setExpression('joy', 1.0)` changes face
- [ ] "Hold to Talk" button turns red when pressed
- [ ] Speech recognition captures voice (console shows transcript)
- [ ] WebSocket sends `{type: 'user_input', text: '...'}`
- [ ] WebSocket receives emotion message
- [ ] Character expression changes to match emotion
- [ ] WebSocket receives audio message
- [ ] Audio plays with character's voice
- [ ] Mouth animates during speech
- [ ] Status returns to "Ready" after response

**If ALL checked:** System working perfectly! 🎉

**If ANY failed:** Follow specific issue debug steps above

---

## 💡 Quick Fixes

### Reset Everything:
```bash
# Kill all background processes
# Restart server
cd F:\Ani
python main_full.py

# In browser:
# Hard refresh: Ctrl+Shift+R
# Clear cache: Ctrl+Shift+Delete
```

### Test with Simple Text (Bypass Voice):
```javascript
// Skip speech recognition, send text directly
ani.sendText('你好')
```

### Force Expression Update:
```javascript
// If expressions stuck, force reset
ani.vrm().expressionManager.setValue('happy', 0)
ani.vrm().expressionManager.setValue('sad', 0)
ani.vrm().expressionManager.setValue('angry', 0)
ani.vrm().expressionManager.setValue('surprised', 0)
ani.vrm().expressionManager.setValue('neutral', 0)

// Then set target
ani.setExpression('joy', 1.0)
```

---

## 📝 Reporting Issues

If still not working, collect this info:

1. **Browser Console Output** (F12 → Console → copy all)
2. **Server Console Output** (last 50 lines)
3. **/debug page output** (copy expression list)
4. **Network → WS messages** (screenshot)
5. **Specific error messages**

This will help identify the exact issue!
