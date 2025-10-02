"""
Audio test client for Ani WebSocket server
Sends synthetic audio and measures VAD latency
"""
import asyncio
import json
import websockets
import numpy as np
import struct


async def test_audio_vad():
    """Test VAD with synthetic audio"""
    uri = "ws://localhost:8000/ws"

    async with websockets.connect(uri) as websocket:
        print("Connected to Ani server\n")

        # Audio config (Silero VAD requires EXACTLY 512 samples at 16kHz)
        sample_rate = 16000
        chunk_size = 512  # Fixed size for Silero VAD
        chunk_duration_ms = (chunk_size / sample_rate) * 1000  # ~32ms

        # Test 1: Send silence (should detect no speech)
        print("Test 1: Sending silence...")
        silence = np.zeros(chunk_size, dtype=np.float32)
        silence_bytes = (silence * 32768).astype(np.int16).tobytes()

        await websocket.send(silence_bytes)
        response = await websocket.recv()
        print(f"Response: {response}\n")

        # Test 2: Send speech-like audio (sine wave)
        print("Test 2: Sending speech-like audio (440Hz tone)...")
        t = np.linspace(0, chunk_duration_ms / 1000, chunk_size)
        speech_audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        speech_bytes = (speech_audio * 32768).astype(np.int16).tobytes()

        await websocket.send(speech_bytes)
        response = await websocket.recv()
        print(f"Response: {response}\n")

        # Test 3: Multiple chunks
        print("Test 3: Sending 10 chunks and measuring average VAD latency...")
        latencies = []

        for i in range(10):
            # Alternate between speech and silence
            t = np.linspace(0, chunk_duration_ms / 1000, chunk_size)
            if i % 2 == 0:
                audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
            else:
                audio = np.zeros(chunk_size, dtype=np.float32)

            audio_bytes = (audio * 32768).astype(np.int16).tobytes()

            await websocket.send(audio_bytes)
            response = await websocket.recv()

            result = json.loads(response)
            if "latency_ms" in result:
                latencies.append(result["latency_ms"])

            print(f"  Chunk {i+1}: VAD={result.get('is_speech', 'N/A')}, "
                  f"prob={result.get('speech_prob', 0):.2f}, "
                  f"latency={result.get('latency_ms', 0):.1f}ms")

        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)

            print(f"\nVAD Latency Stats:")
            print(f"  Average: {avg_latency:.1f}ms")
            print(f"  Min: {min_latency:.1f}ms")
            print(f"  Max: {max_latency:.1f}ms")
            print(f"  Target: <150ms")

            if avg_latency < 150:
                print(f"  [OK] PASSED (within target)")
            else:
                print(f"  [WARN] EXCEEDED (above target)")


async def test_json_and_audio():
    """Test mixing JSON and audio messages"""
    uri = "ws://localhost:8000/ws"

    async with websockets.connect(uri) as websocket:
        print("\nTest 4: Mixing JSON and audio messages...")

        # Send JSON
        test_msg = {"type": "test", "message": "Hello!"}
        await websocket.send(json.dumps(test_msg))
        response = await websocket.recv()
        print(f"JSON response: {response}")

        # Send audio (Silero VAD requires exactly 512 samples at 16kHz)
        sample_rate = 16000
        chunk_size = 512
        duration = chunk_size / sample_rate
        t = np.linspace(0, duration, chunk_size)
        audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        audio_bytes = (audio * 32768).astype(np.int16).tobytes()

        await websocket.send(audio_bytes)
        response = await websocket.recv()
        print(f"Audio response: {response}")

        print("[OK] Mixed message types work")


if __name__ == "__main__":
    print("Ani Audio Pipeline Test Client")
    print("Make sure the server is running (python main_audio.py)\n")
    try:
        asyncio.run(test_audio_vad())
        asyncio.run(test_json_and_audio())
        print("\n[OK] All audio tests completed!")
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        print("Make sure:")
        print("  1. Server is running (python main_audio.py)")
        print("  2. Audio dependencies installed (pip install -r requirements.txt)")
