# Install Ollama - 3 Easy Steps (5 minutes)

## Why Install Ollama?

Right now Ani uses **Mock responses** (scripted, limited variety).
With Ollama, Ani uses **real AI** on your GPU - much smarter and natural!

---

## Step 1: Download Ollama (30 seconds)

Click this link to download:
### 👉 https://ollama.com/download/windows

Or search "Ollama download Windows" in your browser.

The file is called: **OllamaSetup.exe** (~600MB)

---

## Step 2: Install Ollama (2 minutes)

1. **Run** the downloaded `OllamaSetup.exe`
2. **Click** "Install" (it's automatic)
3. **Wait** for it to finish (~1-2 minutes)
4. Ollama will start automatically

That's it! Ollama is now running in the background.

---

## Step 3: Download the AI Model (2-5 minutes)

1. **Open PowerShell** (search for it in Start menu)

2. **Run this command**:
   ```powershell
   ollama pull llama3.1:8b
   ```

3. **Wait** while it downloads (~4.7GB)
   - Fast internet: 2-3 minutes
   - Slow internet: 5-10 minutes

You'll see a progress bar like this:
```
pulling manifest
pulling model (4.7 GB)
███████████████████░░░░░░  75%
```

4. When done, you'll see:
   ```
   success
   ```

---

## Step 4: Restart Ani (10 seconds)

1. **Go back to your terminal** where Ani is running

2. **Press Ctrl+C** to stop the server

3. **Start it again**:
   ```bash
   python main_full.py
   ```

4. **Look for this message**:
   ```
   [OK] Ollama backend initialized (llama3.1:8b)
   ```

   Instead of:
   ```
   [WARN] Ollama not available, falling back to Mock
   ```

---

## You're Done! 🎉

Now when you talk to Ani:
- ✅ Uses your **RTX 4060 GPU**
- ✅ **Real AI** responses (not scripted!)
- ✅ **Much smarter** and more natural
- ✅ **Still fast** (50-200ms with GPU)
- ✅ **100% free** and local

---

## Verify It's Working

1. Open browser: http://localhost:8000
2. Click microphone and say: **"Tell me about yourself"**
3. Ani should give a detailed, unique response!

If she still sounds scripted, check the terminal - you should see:
```
[OK] Ollama backend initialized (llama3.1:8b)
```

---

## Troubleshooting

### "ollama: command not found"

**Solution**: Restart your terminal after installing Ollama, or reboot your computer.

### Download is slow

**Solution**: Be patient! It's downloading 4.7GB. You can continue using the Mock version while it downloads.

### Still using Mock after installation

**Solution**:
1. Make sure you ran `ollama pull llama3.1:8b`
2. Check Ollama is running: `ollama list` (should show llama3.1:8b)
3. Restart Ani server

---

## Need Help?

If you have issues, Ani still works great with the improved Mock responses!

Ollama just makes her even better. 😊

---

**Time to complete**: 5 minutes
**Download size**: ~5.3GB total (Ollama 600MB + Model 4.7GB)
**Benefit**: MUCH smarter Ani with your GPU!
