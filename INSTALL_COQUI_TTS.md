# Installing Coqui TTS for Natural Anime Voice

Coqui TTS with XTTS-v2 will give Ani a beautiful, natural-sounding anime voice instead of the robotic Edge TTS!

## Features
- ✅ **Natural sounding** - Sounds like a real person, not robotic
- ✅ **Voice cloning** - Use any anime voice from a 3-6 second sample
- ✅ **GPU accelerated** - Uses your RTX 4060
- ✅ **Multilingual** - Supports 16 languages including Japanese
- ✅ **Open source** - Completely free

---

## Step 1: Install Coqui TTS

Open **PowerShell** and run:

```powershell
python -m pip install TTS
```

This will install Coqui TTS and all its dependencies (~500MB download).

---

## Step 2: Download the XTTS-v2 Model

The model will download automatically on first use (~2GB). To pre-download it, run:

```powershell
cd F:\Ani
python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"
```

Wait for the download to complete (may take 5-10 minutes).

---

## Step 3: (Optional) Get an Anime Voice Sample

To use voice cloning for a specific anime character:

1. Find a 3-6 second audio clip of your favorite anime character
2. Save it as `F:\Ani\voice_samples\ani_voice.wav`
3. The voice should be clear with minimal background noise

**Example sources:**
- Extract from anime episodes
- Find voice lines from games
- Use YouTube clips (download with yt-dlp)

**Recommended format:**
- WAV or MP3
- 22050 Hz sample rate
- Mono or stereo
- 3-6 seconds long

---

## Step 4: Update Ani Configuration

The code has been updated to support Coqui TTS. To enable it:

1. Open `F:\Ani\main_full.py`
2. Find this line (around line 102):
   ```python
   tts_config = TTSConfig(engine="edge", voice="en-US-AriaNeural")
   ```
3. Change it to:
   ```python
   tts_config = TTSConfig(
       engine="coqui",
       voice="tts_models/multilingual/multi-dataset/xtts_v2",
       speaker_wav="voice_samples/ani_voice.wav"  # Optional: your anime voice sample
   )
   ```

---

## Step 5: Restart Ani

Stop the current server (Ctrl+C) and start it again:

```powershell
cd F:\Ani
python main_full.py
```

Look for this message:
```
[OK] Coqui TTS initialized (xtts_v2)
```

---

## Comparison: Edge TTS vs Coqui TTS

| Feature | Edge TTS (Current) | Coqui TTS (New) |
|---------|-------------------|-----------------|
| **Sound Quality** | Robotic, Microsoft voice | Natural, human-like |
| **Voice Cloning** | ❌ No | ✅ Yes |
| **Latency** | ~500ms (cloud) | ~1-2s (local GPU) |
| **Internet Required** | ✅ Yes | ❌ No (runs offline!) |
| **Anime Voice** | ❌ Generic | ✅ Any anime character! |
| **Cost** | Free | Free |
| **GPU Usage** | None | Yes (RTX 4060) |

---

## Troubleshooting

### Error: "No module named 'TTS'"
```powershell
python -m pip install --upgrade pip
python -m pip install TTS
```

### Error: "CUDA out of memory"
The model needs ~2GB VRAM. Close other GPU applications.

### Voice sounds different than expected
- Try a different voice sample (3-6 seconds, clear audio)
- Adjust the speaker_wav path in the config
- Make sure the sample is high quality

### Slow generation (>5 seconds)
First generation is always slow. Subsequent generations should be ~1-2 seconds.

---

## Alternative: Use Pre-made Anime Voices

If you don't have a voice sample, Coqui TTS can still generate natural voices. Just use:

```python
tts_config = TTSConfig(
    engine="coqui",
    voice="tts_models/multilingual/multi-dataset/xtts_v2"
)
```

This will use the default female voice, which sounds much better than Edge TTS!

---

## Next Steps

After installation, try talking to Ani:

1. **Start Ani**: `python main_full.py`
2. **Open browser**: http://localhost:8000
3. **Click microphone** and say: "Hello Ani, how do you sound now?"
4. **Listen** to the beautiful natural voice! 🎉

---

## Tips for Best Results

- **Clear samples**: Use clean audio without background music
- **Match gender**: Female samples work best for Ani
- **Length matters**: 3-6 seconds is optimal (not too short, not too long)
- **Format**: WAV files work best, but MP3 is fine too
- **Sample rate**: 22050 Hz is ideal

---

## Example Voice Samples to Try

Create a folder `F:\Ani\voice_samples\` and add:

- `ani_voice.wav` - Your main anime character voice
- `backup_voice.wav` - Alternative voice if needed

The TTS pipeline will automatically use them!

---

**Need help? Check the logs for detailed error messages when starting Ani.**
