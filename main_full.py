"""
Ani v0 - Complete Voice Companion Server
Full pipeline: Voice Input → VAD → STT → LLM → TTS → Voice Output
"""
import asyncio
import json
import time
from typing import Dict, Optional, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
import uvicorn

from audio_pipeline import AudioPipeline, AudioConfig
from llm_pipeline import LLMPipeline, LLMConfig
from tts_pipeline import TTSPipeline, TTSConfig


# JSON Schema Models
class Emote(BaseModel):
    type: str = Field(..., pattern="^(joy|sad|anger|surprise|neutral)$")
    intensity: float = Field(..., ge=0.0, le=1.0)


class LLMResponse(BaseModel):
    utterance: str = Field(..., max_length=500)
    emote: Emote
    intent: str = Field(..., pattern="^(SMALL_TALK|ANSWER|ASK|JOKE|TOOL_USE)$")
    phoneme_hints: List[List] = Field(default_factory=list)

    @field_validator('phoneme_hints')
    @classmethod
    def validate_phoneme_hints(cls, v):
        for hint in v:
            if len(hint) != 3:
                raise ValueError("Each phoneme hint must have [phoneme, start_ms, end_ms]")
        return v


# Latency tracking
class LatencyMetrics:
    def __init__(self):
        self.metrics: Dict[str, list] = {
            "vad": [],
            "stt": [],
            "llm": [],
            "tts": [],
            "total": []
        }

    def add_metric(self, stage: str, latency_ms: float):
        if stage in self.metrics:
            self.metrics[stage].append(latency_ms)
            if len(self.metrics[stage]) > 100:
                self.metrics[stage].pop(0)

    def get_stats(self, stage: str) -> dict:
        if not self.metrics.get(stage):
            return {"avg": 0, "min": 0, "max": 0, "count": 0}
        values = self.metrics[stage]
        return {
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "count": len(values)
        }


# FastAPI app
app = FastAPI(title="Ani v0 - Complete Voice Companion")
metrics = LatencyMetrics()

# Global pipelines
audio_pipeline: Optional[AudioPipeline] = None
llm_pipeline: Optional[LLMPipeline] = None
tts_pipeline: Optional[TTSPipeline] = None


@app.on_event("startup")
async def startup_event():
    """Initialize all pipelines on startup"""
    global audio_pipeline, llm_pipeline, tts_pipeline

    print("=" * 60)
    print("Initializing Ani v0 - Complete Voice Companion")
    print("=" * 60)

    # Initialize LLM pipeline
    try:
        # Choose LLM backend:
        # Option 1: Ollama (local, free) - using Qwen2.5 for Chinese support
        llm_config = LLMConfig(
            backend="ollama",
            model="qwen2.5:7b",  # Bilingual EN+ZH model
            character_name="Ani",
            character_personality="friendly and cheerful anime companion"
        )

        # Option 2: OpenAI (GPT-4, GPT-3.5-turbo) - requires API key with credits
        # llm_config = LLMConfig(
        #     backend="openai",
        #     model="gpt-4o-mini",
        #     openai_api_key="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",
        #     character_name="Ani",
        #     character_personality="friendly and cheerful anime companion"
        # )

        llm_pipeline = LLMPipeline(llm_config)
        await llm_pipeline.initialize()
    except Exception as e:
        print(f"[WARN] LLM initialization failed: {e}")

    # Initialize TTS pipeline
    try:
        # Choose TTS engine:
        # - "edge": Fast, robotic Microsoft voice (default, always works)
        # - "coqui": Natural anime voice (requires: pip install TTS)

        # Option 1: Edge TTS (current - robotic voice)
        # tts_config = TTSConfig(engine="edge", voice="en-US-AriaNeural")

        # Option 2: Coqui XTTS-v2 (voice cloning, multi-lingual EN+ZH)
        tts_config = TTSConfig(
            engine="coqui",
            voice="tts_models/multilingual/multi-dataset/xtts_v2",
            speaker_wav="voice_samples/wzy.wav"  # Anime-style voice sample
        )

        tts_pipeline = TTSPipeline(tts_config)
        await tts_pipeline.initialize()
    except Exception as e:
        print(f"[WARN] TTS initialization failed: {e}")

    # Initialize Audio pipeline
    try:
        audio_config = AudioConfig()
        audio_pipeline = AudioPipeline(audio_config)
        await audio_pipeline.load_models()
    except Exception as e:
        print(f"[WARN] Audio pipeline initialization failed: {e}")

    print("=" * 60)
    print("[OK] Ani v0 Server Ready!")
    print("=" * 60)


@app.get("/")
async def root():
    """Serve the frontend"""
    return FileResponse("frontend/index.html")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "pipelines": {
            "audio": audio_pipeline.vad.is_loaded if audio_pipeline else False,
            "llm": llm_pipeline.is_ready if llm_pipeline else False,
            "tts": tts_pipeline.is_ready if tts_pipeline else False
        },
        "latency_stats": {
            stage: metrics.get_stats(stage)
            for stage in metrics.metrics.keys()
        }
    }


@app.post("/api/synthesize")
async def synthesize_speech(request: Request):
    """Synthesize speech from text using Coqui TTS"""
    try:
        body = await request.json()
        text = body.get("text", "")

        if not text:
            return Response(content=b"", status_code=400)

        if not tts_pipeline or not tts_pipeline.is_ready:
            return Response(content=b"", status_code=503)

        # Generate speech using Coqui TTS
        result = await tts_pipeline.synthesize_with_phonemes(text)
        audio_bytes = result["audio"]

        # Return audio as WAV
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"inline; filename=speech.wav"
            }
        )
    except Exception as e:
        print(f"[FAIL] TTS synthesis error: {e}")
        return Response(content=b"", status_code=500)


@app.get("/metrics")
async def get_metrics():
    """Get detailed latency metrics"""
    return {
        stage: {
            **metrics.get_stats(stage),
            "recent": metrics.metrics[stage][-10:] if metrics.metrics[stage] else []
        }
        for stage in metrics.metrics.keys()
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for voice interaction
    Supports: text input, JSON messages
    """
    await websocket.accept()
    print("[Client] Connected")

    try:
        while True:
            try:
                message = await websocket.receive()
            except RuntimeError:
                # WebSocket disconnected
                break

            if "text" in message:
                data = message["text"]

                try:
                    json_msg = json.loads(data)

                    # Handle user text input
                    if json_msg.get("type") == "user_input":
                        user_text = json_msg.get("text", "")

                        if not user_text:
                            continue

                        print(f"[User] {user_text}")

                        # Process through LLM
                        total_start = time.time()

                        if llm_pipeline and llm_pipeline.is_ready:
                            # Generate LLM response
                            llm_start = time.time()
                            llm_response = await llm_pipeline.generate_response(user_text)
                            llm_latency = (time.time() - llm_start) * 1000
                            metrics.add_metric("llm", llm_latency)

                            print(f"[Ani] {llm_response['utterance']}")
                            print(f"[Emote] {llm_response['emote']['type']} ({llm_response['emote']['intensity']})")
                            print(f"[LLM Latency] {llm_latency:.0f}ms")

                            # Send LLM response to client
                            response = {
                                "status": "success",
                                "validated": True,
                                "data": {
                                    "utterance": llm_response["utterance"],
                                    "emote": llm_response["emote"],
                                    "intent": llm_response["intent"],
                                    "phoneme_hints": llm_response.get("phoneme_hints", [])
                                },
                                "llm_latency_ms": llm_latency,
                                "total_latency_ms": (time.time() - total_start) * 1000
                            }

                            await websocket.send_json(response)

                        else:
                            # Fallback response
                            await websocket.send_json({
                                "status": "error",
                                "error": "LLM pipeline not ready"
                            })

                    # Handle generic JSON (for testing)
                    elif all(k in json_msg for k in ["utterance", "emote", "intent"]):
                        validated = LLMResponse(**json_msg)
                        await websocket.send_json({
                            "status": "success",
                            "validated": True,
                            "data": validated.model_dump()
                        })

                    else:
                        # Echo back unknown messages
                        await websocket.send_json({
                            "status": "success",
                            "validated": False,
                            "echo": json_msg
                        })

                except json.JSONDecodeError:
                    await websocket.send_json({
                        "status": "error",
                        "error": "Invalid JSON"
                    })
                except Exception as e:
                    await websocket.send_json({
                        "status": "error",
                        "error": str(e),
                        "type": type(e).__name__
                    })

    except WebSocketDisconnect:
        print("[Client] Disconnected")


if __name__ == "__main__":
    print("Starting Ani v0 - Complete Voice Companion Server...")
    print("Open your browser to: http://localhost:8000")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
