# Ollama Setup Guide for Ani v0

## What is Ollama?

Ollama is a tool that runs large language models (LLMs) locally on your machine. For Ani, we use it to run Llama 3.1 8B for generating character responses.

## Installation

### Windows

1. **Download Ollama**
   - Go to: https://ollama.com/download/windows
   - Download and run the installer
   - Or use command line:
   ```powershell
   winget install Ollama.Ollama
   ```

2. **Verify Installation**
   ```bash
   ollama --version
   ```

3. **Download Llama 3.1 8B Model**
   ```bash
   ollama pull llama3.1:8b
   ```

   This will download ~4.7GB. The model will be cached locally.

4. **Test Ollama**
   ```bash
   ollama run llama3.1:8b "Hello!"
   ```

## GPU Acceleration

Ollama automatically uses your NVIDIA GPU if available. To verify:

```bash
ollama run llama3.1:8b "Test" --verbose
```

You should see GPU being used. With RTX 4060 (8GB VRAM), Llama 3.1 8B runs very fast!

## Using Ollama with Ani

Once Ollama is installed and running:

1. **Start Ollama service** (usually auto-starts on Windows)
   ```bash
   # Check if running
   curl http://localhost:11434/api/tags
   ```

2. **Run Ani server**
   ```bash
   python main_audio.py
   ```

   Ani will automatically detect Ollama and use it for LLM responses.

## Ollama API Endpoints

Ani uses these Ollama API endpoints:

- **List models**: `GET http://localhost:11434/api/tags`
- **Generate**: `POST http://localhost:11434/api/generate`

## Expected Performance

With RTX 4060 (8GB VRAM):
- **Model load**: ~2-5 seconds (first time)
- **Inference latency**: 50-200ms per response
- **Target**: <400ms ✓

## Troubleshooting

### Ollama not found
- Make sure Ollama is installed
- Check if service is running: `ollama list`
- Restart Ollama service

### GPU not being used
- Update NVIDIA drivers
- Reinstall Ollama
- Check CUDA is available: `nvidia-smi`

### Model not found
- Download model: `ollama pull llama3.1:8b`
- List installed models: `ollama list`

### Port 11434 in use
- Default port is 11434
- Change in Ani config if needed:
  ```python
  config = LLMConfig(ollama_host="http://localhost:CUSTOM_PORT")
  ```

## Alternative: Mock LLM (No Ollama Required)

If you don't want to install Ollama, Ani automatically falls back to a Mock LLM backend:
- ✓ Fast (<110ms latency)
- ✓ Deterministic responses
- ✗ Not very intelligent (scripted responses)

Good for testing the pipeline without Ollama.

## Model Recommendations

| Model | Size | VRAM | Speed | Quality |
|-------|------|------|-------|---------|
| llama3.1:8b | 4.7GB | 6-8GB | Fast | High |
| llama3.2:3b | 2.0GB | 3-4GB | Faster | Medium |
| phi3:mini | 2.2GB | 3-4GB | Fastest | Medium |

**Recommended**: `llama3.1:8b` for RTX 4060

## Changing Models

Edit `llm_pipeline.py`:

```python
config = LLMConfig(
    backend="ollama",
    model="llama3.1:8b",  # Change this
    ...
)
```

Available models: https://ollama.com/library

## JSON Mode

Ani uses Ollama's JSON mode to enforce structured output:

```json
{
  "utterance": "Hello! I'm Ani!",
  "emote": {"type": "joy", "intensity": 0.8},
  "intent": "SMALL_TALK",
  "phoneme_hints": []
}
```

This ensures LLM responses always match the required schema.

## Resources

- Ollama website: https://ollama.com
- Ollama docs: https://github.com/ollama/ollama
- Model library: https://ollama.com/library
- Discord community: https://discord.gg/ollama

## Next Steps

After installing Ollama:
1. Test LLM pipeline: `python llm_pipeline.py`
2. Run full server: `python main_audio.py`
3. Test end-to-end: Send text → Get AI response
