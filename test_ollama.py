import asyncio
import aiohttp
import time
import json

async def test_ollama():
    print("Testing Ollama generation...")

    prompt = """You are Ani, a friendly, enthusiastic anime companion.

User said: "hello"

You must respond with valid JSON matching this exact format:
{
  "utterance": "your response text here (max 500 chars)",
  "emote": {
    "type": "joy|sad|anger|surprise|neutral",
    "intensity": 0.0-1.0
  },
  "intent": "SMALL_TALK|ANSWER|ASK|JOKE|TOOL_USE",
  "phoneme_hints": []
}

Respond as Ani in JSON format:"""

    payload = {
        "model": "llama3.1:8b",
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.7,
            "num_predict": 500,
        }
    }

    print("Sending request to Ollama...")
    print(f"Timeout: 30 seconds")

    start = time.time()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                print(f"Status: {resp.status}")

                if resp.status == 200:
                    result = await resp.json()
                    response_text = result.get("response", "")

                    elapsed = time.time() - start
                    print(f"\nElapsed time: {elapsed:.2f}s")
                    print(f"\nRaw response:\n{response_text}\n")

                    try:
                        parsed = json.loads(response_text)
                        print(f"Parsed JSON:")
                        print(json.dumps(parsed, indent=2))
                        print("\n✅ SUCCESS!")
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON parsing failed: {e}")
                        print("Response is not valid JSON")
                else:
                    print(f"❌ HTTP error: {resp.status}")
                    text = await resp.text()
                    print(f"Response: {text}")

    except asyncio.TimeoutError:
        elapsed = time.time() - start
        print(f"\n❌ TIMEOUT after {elapsed:.2f}s")
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n❌ ERROR after {elapsed:.2f}s: {e}")

if __name__ == "__main__":
    asyncio.run(test_ollama())
