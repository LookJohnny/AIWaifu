"""
Ani v0 - Real-time Anime Companion with Audio Pipeline
Extended WebSocket server with VAD + STT integration
"""
import asyncio
import json
import time
from typing import Dict, Optional, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import uvicorn
import numpy as np

from audio_pipeline import AudioPipeline, AudioConfig


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
            if not isinstance(hint[0], str) or not isinstance(hint[1], (int, float)) or not isinstance(hint[2], (int, float)):
                raise ValueError("Phoneme hint format: [str, number, number]")
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
            # Keep only last 100 measurements
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
app = FastAPI(title="Ani v0 - Anime Companion API")
metrics = LatencyMetrics()

# Global audio pipeline
audio_pipeline: Optional[AudioPipeline] = None


@app.on_event("startup")
async def startup_event():
    """Initialize audio pipeline on startup"""
    global audio_pipeline
    print("Initializing audio pipeline...")
    try:
        audio_pipeline = AudioPipeline(AudioConfig())
        await audio_pipeline.load_models()
        print("[OK] Audio pipeline initialized")
    except Exception as e:
        print(f"[WARN] Failed to initialize audio pipeline: {e}")
        print("  Server will run in JSON-only mode")


@app.get("/")
async def root():
    return {
        "message": "Ani v0 - Real-time Anime Companion",
        "status": "running",
        "audio_enabled": audio_pipeline is not None and audio_pipeline.vad.is_loaded
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "audio_pipeline": {
            "vad_loaded": audio_pipeline.vad.is_loaded if audio_pipeline else False,
            "stt_loaded": audio_pipeline.stt.is_loaded if audio_pipeline else False,
        },
        "latency_stats": {
            stage: metrics.get_stats(stage)
            for stage in metrics.metrics.keys()
        }
    }


@app.get("/metrics")
async def get_metrics():
    """Get detailed latency metrics for all pipeline stages"""
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
    WebSocket endpoint for real-time communication
    Supports both JSON messages and binary audio streams
    """
    await websocket.accept()
    print("Client connected")

    try:
        while True:
            # Receive message (text or binary)
            message = await websocket.receive()

            if "text" in message:
                # Handle JSON messages
                start_time = time.time()
                data = message["text"]

                try:
                    # Parse JSON
                    json_msg = json.loads(data)

                    # Validate if it's an LLM response format
                    if all(k in json_msg for k in ["utterance", "emote", "intent"]):
                        validated = LLMResponse(**json_msg)
                        response = {
                            "status": "success",
                            "validated": True,
                            "data": validated.model_dump(),
                            "latency_ms": (time.time() - start_time) * 1000
                        }
                    else:
                        # Echo back other messages
                        response = {
                            "status": "success",
                            "validated": False,
                            "echo": json_msg,
                            "latency_ms": (time.time() - start_time) * 1000
                        }

                    # Track latency
                    latency_ms = (time.time() - start_time) * 1000
                    metrics.add_metric("total", latency_ms)

                    await websocket.send_json(response)

                except json.JSONDecodeError:
                    await websocket.send_json({
                        "status": "error",
                        "error": "Invalid JSON",
                        "received": data
                    })
                except Exception as e:
                    await websocket.send_json({
                        "status": "error",
                        "error": str(e),
                        "type": type(e).__name__
                    })

            elif "bytes" in message:
                # Handle binary audio data
                if not audio_pipeline or not audio_pipeline.vad.is_loaded:
                    await websocket.send_json({
                        "status": "error",
                        "error": "Audio pipeline not initialized"
                    })
                    continue

                audio_bytes = message["bytes"]

                # Process audio chunk
                try:
                    # Convert bytes to numpy array
                    audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                    # Measure VAD latency
                    vad_start = time.time()
                    speech_prob = audio_pipeline.vad.detect_speech(audio_chunk)
                    vad_latency = (time.time() - vad_start) * 1000

                    metrics.add_metric("vad", vad_latency)

                    is_speech = audio_pipeline.vad.is_speech(speech_prob)

                    # Send VAD result
                    await websocket.send_json({
                        "type": "vad",
                        "is_speech": is_speech,
                        "speech_prob": float(speech_prob),
                        "latency_ms": vad_latency
                    })

                    # TODO: Implement full audio streaming pipeline
                    # For now, just send VAD results

                except Exception as e:
                    await websocket.send_json({
                        "status": "error",
                        "error": f"Audio processing error: {str(e)}"
                    })

    except WebSocketDisconnect:
        print("Client disconnected")


if __name__ == "__main__":
    print("Starting Ani v0 Server with Audio Pipeline...")
    print("WebSocket endpoint: ws://localhost:8000/ws")
    print("  - Send JSON for validation")
    print("  - Send binary audio (16kHz, int16) for VAD/STT")
    print("Health check: http://localhost:8000/health")
    print("Metrics: http://localhost:8000/metrics")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
