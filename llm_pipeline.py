"""
LLM Pipeline for Ani v0
Supports multiple LLM backends with strict JSON schema enforcement
Target: <400ms latency
"""
import asyncio
import json
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class LLMConfig:
    """LLM configuration"""
    backend: str = "ollama"  # ollama, openai, anthropic, local
    model: str = "llama3.1:8b"
    temperature: float = 0.7
    max_tokens: int = 500
    timeout: float = 30.0  # seconds (first generation can be slow)

    # Ollama-specific
    ollama_host: str = "http://localhost:11434"

    # OpenAI-specific
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"

    # Character personality
    character_name: str = "Ani"
    character_personality: str = "friendly, enthusiastic anime companion"


class LLMBackend(ABC):
    """Abstract base class for LLM backends"""

    @abstractmethod
    async def generate(self, prompt: str, schema: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate response with optional JSON schema enforcement"""
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if backend is available"""
        pass


class OllamaBackend(LLMBackend):
    """Ollama LLM backend with JSON mode"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.ollama_host

    async def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def generate(self, prompt: str, schema: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate response using Ollama"""
        import aiohttp
        import time

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            }
        }

        # Enable JSON mode if schema provided
        if schema:
            payload["format"] = "json"

        print(f"[Ollama] Sending request (timeout: {self.config.timeout}s)...")
        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as resp:
                    if resp.status != 200:
                        raise Exception(f"Ollama API error: {resp.status}")

                    result = await resp.json()
                    response_text = result.get("response", "")
                    elapsed = time.time() - start_time
                    print(f"[Ollama] Response received in {elapsed:.2f}s")

                    # Parse JSON if schema was requested
                    if schema and response_text:
                        try:
                            return json.loads(response_text)
                        except json.JSONDecodeError:
                            # Fallback: try to extract JSON from text
                            import re
                            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                            if json_match:
                                return json.loads(json_match.group())
                            raise

                    return {"response": response_text}

        except asyncio.TimeoutError:
            raise Exception(f"LLM timeout after {self.config.timeout}s")
        except Exception as e:
            raise Exception(f"Ollama generation failed: {e}")


class OpenAIBackend(LLMBackend):
    """OpenAI API backend (GPT-4, GPT-3.5-turbo, etc.)"""

    def __init__(self, config: LLMConfig):
        self.config = config
        if not config.openai_api_key:
            raise ValueError("OpenAI API key is required. Set openai_api_key in LLMConfig")

    async def is_available(self) -> bool:
        """Check if OpenAI API is accessible"""
        try:
            import aiohttp
            headers = {"Authorization": f"Bearer {self.config.openai_api_key}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config.openai_base_url}/models",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def generate(self, prompt: str, schema: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate response using OpenAI API"""
        import aiohttp
        import time

        headers = {
            "Authorization": f"Bearer {self.config.openai_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": f"You are {self.config.character_name}, a {self.config.character_personality}."},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        # Enable JSON mode if schema provided
        if schema:
            payload["response_format"] = {"type": "json_object"}

        try:
            print(f"[OpenAI] Sending request (model: {self.config.model})...")
            start_time = time.time()

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.openai_base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise Exception(f"OpenAI API error ({resp.status}): {error_text}")

                    result = await resp.json()
                    response_text = result["choices"][0]["message"]["content"]

                    elapsed = time.time() - start_time
                    print(f"[OpenAI] Response received in {elapsed:.2f}s")

                    # Parse JSON if schema was requested
                    if schema and response_text:
                        try:
                            return json.loads(response_text)
                        except json.JSONDecodeError:
                            import re
                            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                            if json_match:
                                return json.loads(json_match.group())
                            raise

                    return {"response": response_text}

        except asyncio.TimeoutError:
            raise Exception(f"OpenAI timeout after {self.config.timeout}s")
        except Exception as e:
            raise Exception(f"OpenAI generation failed: {e}")


class MockLLMBackend(LLMBackend):
    """Mock LLM for testing (fast, varied responses)"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.conversation_count = 0

        # Many varied responses
        self.responses = {
            "greetings": [
                {"utterance": f"Hello! I'm {config.character_name}, your anime companion! How can I help you today?", "emote": {"type": "joy", "intensity": 0.8}, "intent": "SMALL_TALK"},
                {"utterance": "Hi there! It's great to see you! What's on your mind?", "emote": {"type": "joy", "intensity": 0.9}, "intent": "SMALL_TALK"},
                {"utterance": "Hey! I'm so excited to chat with you!", "emote": {"type": "joy", "intensity": 0.7}, "intent": "SMALL_TALK"},
                {"utterance": "Welcome back! I've been waiting to talk to you!", "emote": {"type": "joy", "intensity": 0.8}, "intent": "SMALL_TALK"},
            ],
            "questions": [
                {"utterance": "That's a really interesting question! From what I know, there are many different perspectives on that.", "emote": {"type": "surprise", "intensity": 0.6}, "intent": "ANSWER"},
                {"utterance": "Hmm, let me think about that... I'd say it depends on how you look at it!", "emote": {"type": "neutral", "intensity": 0.5}, "intent": "ANSWER"},
                {"utterance": "Great question! I love when you ask me things like that. The answer is quite fascinating actually!", "emote": {"type": "joy", "intensity": 0.7}, "intent": "ANSWER"},
                {"utterance": "You know, I've wondered about that too! It's one of those things that makes you think, isn't it?", "emote": {"type": "surprise", "intensity": 0.6}, "intent": "ASK"},
            ],
            "positive": [
                {"utterance": "That's wonderful! I'm so happy to hear that!", "emote": {"type": "joy", "intensity": 0.9}, "intent": "SMALL_TALK"},
                {"utterance": "Amazing! That sounds really exciting!", "emote": {"type": "joy", "intensity": 0.8}, "intent": "SMALL_TALK"},
                {"utterance": "How delightful! You just made my day!", "emote": {"type": "joy", "intensity": 1.0}, "intent": "SMALL_TALK"},
            ],
            "negative": [
                {"utterance": "Oh no, I'm sorry to hear that. Is there anything I can do to help?", "emote": {"type": "sad", "intensity": 0.6}, "intent": "ASK"},
                {"utterance": "That must be difficult. I'm here if you want to talk about it.", "emote": {"type": "sad", "intensity": 0.7}, "intent": "SMALL_TALK"},
            ],
            "thanks": [
                {"utterance": "You're very welcome! I'm always happy to help!", "emote": {"type": "joy", "intensity": 0.8}, "intent": "SMALL_TALK"},
                {"utterance": "No problem at all! That's what I'm here for!", "emote": {"type": "joy", "intensity": 0.7}, "intent": "SMALL_TALK"},
                {"utterance": "Anytime! It makes me happy when I can help you!", "emote": {"type": "joy", "intensity": 0.9}, "intent": "SMALL_TALK"},
            ],
            "default": [
                {"utterance": "I see! Tell me more about that, I'm curious!", "emote": {"type": "neutral", "intensity": 0.5}, "intent": "ASK"},
                {"utterance": "That's interesting! I hadn't thought about it that way.", "emote": {"type": "surprise", "intensity": 0.5}, "intent": "SMALL_TALK"},
                {"utterance": "Oh really? I'd love to hear more about your thoughts on this!", "emote": {"type": "joy", "intensity": 0.6}, "intent": "ASK"},
                {"utterance": "I understand what you mean. It's definitely something worth thinking about!", "emote": {"type": "neutral", "intensity": 0.6}, "intent": "SMALL_TALK"},
            ]
        }

    async def is_available(self) -> bool:
        """Mock is always available"""
        return True

    async def generate(self, prompt: str, schema: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate varied mock response"""
        import random

        # Simulate processing time
        await asyncio.sleep(0.1)

        # Detect intent from prompt
        prompt_lower = prompt.lower()

        # Choose response category
        if any(word in prompt_lower for word in ["hello", "hi", "hey", "greetings"]):
            category = "greetings"
        elif any(word in prompt_lower for word in ["?", "what", "how", "why", "when", "where", "who"]):
            category = "questions"
        elif any(word in prompt_lower for word in ["good", "great", "awesome", "happy", "excited", "love"]):
            category = "positive"
        elif any(word in prompt_lower for word in ["sad", "bad", "terrible", "awful", "hate", "angry"]):
            category = "negative"
        elif any(word in prompt_lower for word in ["thank", "thanks", "appreciate"]):
            category = "thanks"
        else:
            category = "default"

        # Get response list
        responses = self.responses.get(category, self.responses["default"])

        # Pick response (rotate through them)
        response = responses[self.conversation_count % len(responses)]
        self.conversation_count += 1

        # Add phoneme_hints if not present
        if "phoneme_hints" not in response:
            response["phoneme_hints"] = []

        return response


class LLMPipeline:
    """
    LLM pipeline with automatic backend selection and JSON schema enforcement
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.backend: Optional[LLMBackend] = None
        self.is_ready = False

        # JSON schema for Ani responses
        self.response_schema = {
            "type": "object",
            "properties": {
                "utterance": {"type": "string", "maxLength": 500},
                "emote": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["joy", "sad", "anger", "surprise", "neutral"]},
                        "intensity": {"type": "number", "minimum": 0, "maximum": 1}
                    },
                    "required": ["type", "intensity"]
                },
                "intent": {"type": "string", "enum": ["SMALL_TALK", "ANSWER", "ASK", "JOKE", "TOOL_USE"]},
                "phoneme_hints": {"type": "array"}
            },
            "required": ["utterance", "emote", "intent"]
        }

    async def initialize(self):
        """Initialize LLM backend"""
        print(f"Initializing LLM pipeline (backend: {self.config.backend})...")

        try:
            # Try OpenAI
            if self.config.backend == "openai":
                openai = OpenAIBackend(self.config)
                if await openai.is_available():
                    self.backend = openai
                    print(f"[OK] OpenAI backend initialized ({self.config.model})")
                    self.is_ready = True
                    return
                else:
                    print("[WARN] OpenAI API not available, falling back to Mock")

            # Try Ollama
            elif self.config.backend == "ollama":
                ollama = OllamaBackend(self.config)
                if await ollama.is_available():
                    self.backend = ollama
                    print(f"[OK] Ollama backend initialized ({self.config.model})")
                    self.is_ready = True
                    return
                else:
                    print("[WARN] Ollama not available, falling back to Mock")

            # Fallback to mock
            self.backend = MockLLMBackend(self.config)
            print("[OK] Mock LLM backend initialized (fast, deterministic responses)")
            self.is_ready = True

        except Exception as e:
            print(f"[FAIL] LLM initialization failed: {e}")
            # Use mock as last resort
            self.backend = MockLLMBackend(self.config)
            self.is_ready = True
            print("[WARN] Using Mock backend as fallback")

    def _build_prompt(self, user_input: str) -> str:
        """Build prompt with character personality and JSON schema requirements"""
        schema_description = """
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
"""

        prompt = f"""You are {self.config.character_name}, a {self.config.character_personality}.
You can speak both English and Chinese (中文). Respond in the same language the user speaks.

User said: "{user_input}"

{schema_description}

Respond as {self.config.character_name} in JSON format:"""

        return prompt

    async def generate_response(self, user_input: str) -> Dict[str, Any]:
        """
        Generate character response from user input

        Args:
            user_input: User's text input

        Returns:
            Dict with utterance, emote, intent, phoneme_hints
        """
        if not self.is_ready or not self.backend:
            raise RuntimeError("LLM pipeline not initialized")

        start_time = time.time()

        try:
            # Build prompt
            prompt = self._build_prompt(user_input)

            # Generate response
            response = await self.backend.generate(prompt, schema=self.response_schema)

            # Validate response has required fields
            required_fields = ["utterance", "emote", "intent"]
            for field in required_fields:
                if field not in response:
                    raise ValueError(f"Missing required field: {field}")

            # Add phoneme_hints if missing
            if "phoneme_hints" not in response:
                response["phoneme_hints"] = []

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            return {
                **response,
                "llm_latency_ms": latency_ms
            }

        except Exception as e:
            # Fallback response on error
            print(f"[FAIL] LLM generation error: {e}")
            return {
                "utterance": "I'm having trouble thinking right now. Can you try again?",
                "emote": {"type": "sad", "intensity": 0.3},
                "intent": "SMALL_TALK",
                "phoneme_hints": [],
                "llm_latency_ms": (time.time() - start_time) * 1000,
                "error": str(e)
            }


# Testing
async def test_llm_pipeline():
    """Test LLM pipeline with various inputs"""
    print("=" * 60)
    print("Testing LLM Pipeline")
    print("=" * 60)

    config = LLMConfig(
        backend="ollama",  # Will fallback to mock if Ollama not available
        character_name="Ani",
        character_personality="cheerful and helpful anime companion"
    )

    pipeline = LLMPipeline(config)
    await pipeline.initialize()

    test_inputs = [
        "Hello!",
        "How are you today?",
        "What's your favorite anime?",
        "Tell me a joke",
        "Thanks for your help!"
    ]

    print("\nTest Inputs:")
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n[Test {i}] User: {user_input}")

        response = await pipeline.generate_response(user_input)

        print(f"[Response] {response['utterance']}")
        print(f"[Emote] {response['emote']['type']} (intensity: {response['emote']['intensity']})")
        print(f"[Intent] {response['intent']}")
        print(f"[Latency] {response['llm_latency_ms']:.0f}ms")

        if response['llm_latency_ms'] < 400:
            print("[OK] Latency within target (<400ms)")
        else:
            print("[WARN] Latency exceeds target (>400ms)")

    # Calculate average latency
    print("\n" + "=" * 60)
    print("LLM Pipeline Tests Complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_llm_pipeline())
