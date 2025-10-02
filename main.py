"""
Ani v0 - Real-time Anime Companion
Main WebSocket server with JSON schema validation
"""
import asyncio
import json
import time
from typing import Dict, Optional, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import uvicorn


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


@app.get("/")
async def root():
    return {"message": "Ani v0 - Real-time Anime Companion", "status": "running"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
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
    Accepts JSON messages and echoes them back with latency info
    """
    await websocket.accept()
    print("Client connected")

    try:
        while True:
            # Receive message
            start_time = time.time()
            data = await websocket.receive_text()

            try:
                # Parse JSON
                message = json.loads(data)

                # Validate if it's an LLM response format
                if all(k in message for k in ["utterance", "emote", "intent"]):
                    validated = LLMResponse(**message)
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
                        "echo": message,
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

    except WebSocketDisconnect:
        print("Client disconnected")


if __name__ == "__main__":
    print("Starting Ani v0 Server...")
    print("WebSocket endpoint: ws://localhost:8000/ws")
    print("Health check: http://localhost:8000/health")
    print("Metrics: http://localhost:8000/metrics")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
