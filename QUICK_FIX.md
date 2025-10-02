# Quick Fix for Ani

## Problem: Ani repeats herself and sounds robotic

**Cause**: Using Mock LLM (scripted responses only)

## Solution 1: Install Ollama (5 minutes - RECOMMENDED)

### Download & Install:
1. Go to: https://ollama.com/download/windows
2. Download the installer
3. Run it (takes 2 minutes)

### Download AI Model:
Open PowerShell and run:
```powershell
ollama pull llama3.1:8b
```
(This downloads ~4.7GB - takes 2-5 minutes depending on internet)

### Restart Ani:
```bash
# Stop the current server (Ctrl+C)
# Start it again:
python main_full.py
```

Now Ani will be MUCH smarter! Uses your GPU for fast responses.

---

## Solution 2: Use Improved Mock (Instant - Good enough for testing)

I've already improved the Mock LLM to be better!

Just:
```bash
# Stop server (Ctrl+C)
# Start again:
python main_full.py
```

The new Mock has varied responses instead of repeating.

---

## Verify It's Working

After either fix, you should see in the terminal:
- With Ollama: `[OK] Ollama backend initialized (llama3.1:8b)`
- With Mock: `[OK] Mock LLM backend initialized (fast, deterministic responses)`

Then she won't repeat anymore!
