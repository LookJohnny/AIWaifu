"""
Test client for Ani WebSocket server
Tests JSON validation and echo functionality
"""
import asyncio
import json
import websockets


async def test_websocket():
    """Test the WebSocket server with various message types"""
    uri = "ws://localhost:8000/ws"

    async with websockets.connect(uri) as websocket:
        print("Connected to Ani server\n")

        # Test 1: Simple echo
        print("Test 1: Simple echo message")
        test_msg = {"type": "test", "message": "Hello Ani!"}
        await websocket.send(json.dumps(test_msg))
        response = await websocket.recv()
        print(f"Response: {response}\n")

        # Test 2: Valid LLM response
        print("Test 2: Valid LLM response")
        llm_response = {
            "utterance": "Hello! I'm Ani, your anime companion. How can I help you today?",
            "emote": {"type": "joy", "intensity": 0.8},
            "intent": "SMALL_TALK",
            "phoneme_hints": [
                ["HH", 0, 50],
                ["AH", 50, 150],
                ["L", 150, 200]
            ]
        }
        await websocket.send(json.dumps(llm_response))
        response = await websocket.recv()
        print(f"Response: {response}\n")

        # Test 3: Invalid emote type
        print("Test 3: Invalid emote type (should fail)")
        invalid_msg = {
            "utterance": "Test",
            "emote": {"type": "invalid", "intensity": 0.5},
            "intent": "SMALL_TALK"
        }
        await websocket.send(json.dumps(invalid_msg))
        response = await websocket.recv()
        print(f"Response: {response}\n")

        # Test 4: Invalid intensity
        print("Test 4: Invalid intensity (should fail)")
        invalid_intensity = {
            "utterance": "Test",
            "emote": {"type": "joy", "intensity": 1.5},
            "intent": "SMALL_TALK"
        }
        await websocket.send(json.dumps(invalid_intensity))
        response = await websocket.recv()
        print(f"Response: {response}\n")

        # Test 5: Missing required field
        print("Test 5: Missing required field (should fail)")
        missing_field = {
            "utterance": "Test",
            "emote": {"type": "joy", "intensity": 0.5}
            # missing intent
        }
        await websocket.send(json.dumps(missing_field))
        response = await websocket.recv()
        print(f"Response: {response}\n")


if __name__ == "__main__":
    print("Ani WebSocket Test Client")
    print("Make sure the server is running (python main.py)\n")
    try:
        asyncio.run(test_websocket())
        print("\nAll tests completed!")
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure the server is running on localhost:8000")
