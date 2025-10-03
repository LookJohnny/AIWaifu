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
from animation_controller import AnimationController


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
animation_controller: Optional[AnimationController] = None


@app.on_event("startup")
async def startup_event():
    """Initialize all pipelines on startup"""
    global audio_pipeline, llm_pipeline, tts_pipeline, animation_controller

    print("=" * 60)
    print("Initializing Ani v0 - Complete Voice Companion")
    print("=" * 60)

    # Initialize LLM pipeline
    try:
        # Choose LLM backend:
        # Option 1: Claude 3.5 Haiku (fast, intelligent, bilingual) ✨ ACTIVE
        # Note: Your API key doesn't have access to Sonnet, using Haiku instead
        import os
        llm_config = LLMConfig(
            backend="anthropic",
            model="claude-3-5-haiku-20241022",
            max_tokens=150,  # Shorter responses = faster (default: 500)
            temperature=0.8,  # More creative
            openai_api_key=os.getenv("CLAUDE_API_KEY", "your-api-key-here"),  # Set via environment variable
            character_name="Ani",
            character_personality="friendly and cheerful anime companion"
        )

        # Option 2: Mock (fast, for testing 3D avatar)
        # llm_config = LLMConfig(
        #     backend="mock",
        #     model="mock",
        #     character_name="Ani",
        #     character_personality="friendly and cheerful anime companion"
        # )

        # Option 3: Ollama (local, free) - using Qwen2.5 for Chinese support
        # llm_config = LLMConfig(
        #     backend="ollama",
        #     model="qwen2.5:7b",  # Bilingual EN+ZH model
        #     character_name="Ani",
        #     character_personality="friendly and cheerful anime companion"
        # )

        # Option 4: OpenAI (GPT-4, GPT-3.5-turbo) - requires API key with credits
        # llm_config = LLMConfig(
        #     backend="openai",
        #     model="gpt-4o-mini",
        #     openai_api_key="YOUR_OPENAI_KEY_HERE",
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
        # - "edge": Fast (<1s), natural Microsoft voice, supports Chinese ✨ FASTEST
        # - "coqui": High-quality (5-7s), voice cloning, custom voice

        # Option 1: Edge TTS - FAST MODE ✨ ACTIVE (少女风格 - 活泼开朗)
        tts_config = TTSConfig(
            engine="edge",
            voice="zh-CN-XiaoyiNeural",  # 少女 - 活泼开朗，青春洋溢
            rate="+8%",  # Slightly faster speech
            pitch="+5Hz"  # Higher pitch for youthful voice
        )

        # Option 2: Coqui XTTS-v2 (SLOW but highest quality voice cloning)
        # tts_config = TTSConfig(
        #     engine="coqui",
        #     voice="tts_models/multilingual/multi-dataset/xtts_v2",
        #     speaker_wav="voice_samples/female_high_clear_1.wav"
        # )

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

    # Initialize Animation Controller (3D character via VSeeFace)
    try:
        animation_controller = AnimationController(host="127.0.0.1", port=39539)
        if animation_controller.connected:
            print("[OK] Animation controller connected to VSeeFace")
        else:
            print("[INFO] VSeeFace not running - animations disabled (voice will work)")
    except Exception as e:
        print(f"[WARN] Animation controller initialization failed: {e}")
        animation_controller = None

    print("=" * 60)
    print("[OK] Ani v0 Server Ready!")
    print("=" * 60)


# Mount static files for VRM model (must be before routes)
app.mount("/character", StaticFiles(directory="character"), name="character")

@app.get("/")
async def root():
    """Serve the complete 3D avatar with all features"""
    return FileResponse("frontend/complete.html")

@app.get("/debug")
async def debug_vrm():
    """VRM expression debugger"""
    return FileResponse("debug_vrm.html")

@app.get("/pose")
async def pose_test():
    """Pose adjustment tool"""
    return FileResponse("frontend/pose_test.html")

@app.get("/pro")
async def avatar_pro():
    """Professional animation system"""
    return FileResponse("frontend/avatar_pro.html")

@app.get("/vrma")
async def avatar_vrma():
    """VRMA professional motion capture animation (experimental)"""
    return FileResponse("frontend/avatar_vrma.html")

@app.get("/stable")
async def avatar_stable():
    """Stable animation with natural pose - uses proven avatar_3d system"""
    return FileResponse("frontend/avatar_3d.html")

@app.get("/fixed")
async def avatar_fixed():
    """FIXED VERSION - Debug and properly working animations"""
    return FileResponse("frontend/avatar_fixed.html")

@app.get("/test")
async def avatar_test():
    """Simple test page for VRM pose debugging"""
    return FileResponse("frontend/test_simple.html")

# Mount animations directory
app.mount("/animations", StaticFiles(directory="animations"), name="animations")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "pipelines": {
            "audio": audio_pipeline.vad.is_loaded if audio_pipeline else False,
            "llm": llm_pipeline.is_ready if llm_pipeline else False,
            "tts": tts_pipeline.is_ready if tts_pipeline else False,
            "animation": animation_controller.connected if animation_controller else False
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

                            # Trigger character expression animation
                            if animation_controller and animation_controller.connected:
                                emotion = llm_response['emote']['type']
                                intensity = llm_response['emote']['intensity']
                                asyncio.create_task(animation_controller.set_expression(emotion, intensity))

                            # Send emotion to frontend
                            await websocket.send_json({
                                "type": "emotion",
                                "emotion": llm_response['emote']['type'],
                                "intensity": llm_response['emote']['intensity']
                            })

                            # Generate and send audio
                            if tts_pipeline and tts_pipeline.is_ready:
                                tts_start = time.time()
                                tts_result = await tts_pipeline.synthesize_with_phonemes(llm_response["utterance"])
                                tts_latency = (time.time() - tts_start) * 1000
                                metrics.add_metric("tts", tts_latency)

                                # Convert audio to base64
                                import base64
                                audio_base64 = base64.b64encode(tts_result["audio"]).decode('utf-8')

                                # Send audio to frontend
                                await websocket.send_json({
                                    "type": "audio",
                                    "audio": audio_base64,
                                    "text": llm_response["utterance"]
                                })

                                print(f"[TTS Latency] {tts_latency:.0f}ms")

                            # Send complete response
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
