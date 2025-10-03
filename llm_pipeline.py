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


class AnthropicBackend(LLMBackend):
    """Anthropic Claude API backend (Claude 3.5 Haiku, Sonnet, Opus)"""

    def __init__(self, config: LLMConfig):
        self.config = config
        if not config.openai_api_key:
            raise ValueError("Anthropic API key is required. Set openai_api_key in LLMConfig")
        self.api_key = config.openai_api_key

    async def is_available(self) -> bool:
        """Check if Anthropic API is accessible"""
        try:
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic(api_key=self.api_key)
            # Test with a minimal request
            await client.messages.create(
                model=self.config.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}]
            )
            return True
        except Exception:
            return False

    async def generate(self, prompt: str, schema: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate response using Anthropic Claude"""
        from anthropic import AsyncAnthropic
        import time

        client = AsyncAnthropic(api_key=self.api_key)

        # Build system prompt with JSON schema requirement
        system_prompt = f"You are {self.config.character_name}, a {self.config.character_personality}."

        if schema:
            system_prompt += "\n\nYou MUST respond with valid JSON matching this exact format:\n"
            system_prompt += """{
  "utterance": "your response text here (max 500 chars)",
  "emote": {
    "type": "joy|sad|anger|surprise|neutral",
    "intensity": 0.0-1.0
  },
  "intent": "SMALL_TALK|ANSWER|ASK|JOKE|TOOL_USE",
  "phoneme_hints": []
}"""

        try:
            print(f"[Claude] Sending request (model: {self.config.model})...")
            start_time = time.time()

            response = await client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            response_text = response.content[0].text
            elapsed = time.time() - start_time
            print(f"[Claude] Response received in {elapsed:.2f}s")

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

        except Exception as e:
            raise Exception(f"Claude generation failed: {e}")


class MockLLMBackend(LLMBackend):
    """Mock LLM for testing (fast, varied responses)"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.conversation_count = 0

        # Bilingual responses (中文 + English)
        self.responses = {
            "greetings_zh": [
                {"utterance": f"你好呀！我是{config.character_name}，很高兴见到你！有什么我可以帮你的吗？", "emote": {"type": "joy", "intensity": 0.9}, "intent": "SMALL_TALK"},
                {"utterance": "嗨！见到你真开心！今天过得怎么样？", "emote": {"type": "joy", "intensity": 0.8}, "intent": "SMALL_TALK"},
                {"utterance": "欢迎欢迎！我一直在等你来聊天呢！", "emote": {"type": "joy", "intensity": 0.85}, "intent": "SMALL_TALK"},
                {"utterance": "你来啦！太好了，我们聊点什么吧！", "emote": {"type": "joy", "intensity": 0.8}, "intent": "SMALL_TALK"},
            ],
            "greetings_en": [
                {"utterance": f"Hello! I'm {config.character_name}! How can I help you today?", "emote": {"type": "joy", "intensity": 0.8}, "intent": "SMALL_TALK"},
                {"utterance": "Hi there! Great to see you! What's on your mind?", "emote": {"type": "joy", "intensity": 0.9}, "intent": "SMALL_TALK"},
                {"utterance": "Hey! I'm so excited to chat with you!", "emote": {"type": "joy", "intensity": 0.7}, "intent": "SMALL_TALK"},
            ],
            "questions_zh": [
                {"utterance": "这个问题很有意思！让我想想……其实有很多不同的角度可以看待这个问题。", "emote": {"type": "neutral", "intensity": 0.6}, "intent": "ANSWER"},
                {"utterance": "哇，好问题！我也很好奇这个呢！", "emote": {"type": "surprise", "intensity": 0.7}, "intent": "ASK"},
                {"utterance": "嗯……我觉得这要看具体情况！你觉得呢？", "emote": {"type": "neutral", "intensity": 0.5}, "intent": "ASK"},
            ],
            "questions_en": [
                {"utterance": "That's a really interesting question! Let me think about that...", "emote": {"type": "surprise", "intensity": 0.6}, "intent": "ANSWER"},
                {"utterance": "Great question! I love when you ask me things like that!", "emote": {"type": "joy", "intensity": 0.7}, "intent": "ANSWER"},
            ],
            "positive_zh": [
                {"utterance": "太好了！我真为你高兴！", "emote": {"type": "joy", "intensity": 0.95}, "intent": "SMALL_TALK"},
                {"utterance": "哇！听起来超棒的！", "emote": {"type": "joy", "intensity": 0.9}, "intent": "SMALL_TALK"},
                {"utterance": "真的吗？太让人开心了！你让我的一天都变美好了！", "emote": {"type": "joy", "intensity": 1.0}, "intent": "SMALL_TALK"},
            ],
            "positive_en": [
                {"utterance": "That's wonderful! I'm so happy for you!", "emote": {"type": "joy", "intensity": 0.9}, "intent": "SMALL_TALK"},
                {"utterance": "Amazing! That sounds really exciting!", "emote": {"type": "joy", "intensity": 0.8}, "intent": "SMALL_TALK"},
            ],
            "negative_zh": [
                {"utterance": "哎呀……听起来有点难过。我能帮你什么吗？", "emote": {"type": "sad", "intensity": 0.7}, "intent": "ASK"},
                {"utterance": "别太难过了，我一直在这里陪着你。", "emote": {"type": "sad", "intensity": 0.6}, "intent": "SMALL_TALK"},
            ],
            "negative_en": [
                {"utterance": "Oh no, I'm sorry to hear that. Is there anything I can do?", "emote": {"type": "sad", "intensity": 0.6}, "intent": "ASK"},
            ],
            "thanks_zh": [
                {"utterance": "不客气！我很乐意帮忙的！", "emote": {"type": "joy", "intensity": 0.8}, "intent": "SMALL_TALK"},
                {"utterance": "没事没事！这是我应该做的！", "emote": {"type": "joy", "intensity": 0.75}, "intent": "SMALL_TALK"},
                {"utterance": "随时都可以！能帮到你我很开心！", "emote": {"type": "joy", "intensity": 0.9}, "intent": "SMALL_TALK"},
            ],
            "thanks_en": [
                {"utterance": "You're very welcome! Always happy to help!", "emote": {"type": "joy", "intensity": 0.8}, "intent": "SMALL_TALK"},
            ],
            "default_zh": [
                {"utterance": "原来如此！跟我说说更多吧，我很好奇！", "emote": {"type": "neutral", "intensity": 0.6}, "intent": "ASK"},
                {"utterance": "有意思！我之前没这样想过呢。", "emote": {"type": "surprise", "intensity": 0.6}, "intent": "SMALL_TALK"},
                {"utterance": "嗯嗯，我明白你的意思！确实值得好好想想。", "emote": {"type": "neutral", "intensity": 0.5}, "intent": "SMALL_TALK"},
                {"utterance": "是这样啊！继续说说吧！", "emote": {"type": "joy", "intensity": 0.6}, "intent": "ASK"},
            ],
            "default_en": [
                {"utterance": "I see! Tell me more, I'm curious!", "emote": {"type": "neutral", "intensity": 0.5}, "intent": "ASK"},
                {"utterance": "That's interesting! I hadn't thought about it that way.", "emote": {"type": "surprise", "intensity": 0.5}, "intent": "SMALL_TALK"},
            ]
        }

    async def is_available(self) -> bool:
        """Mock is always available"""
        return True

    async def generate(self, prompt: str, schema: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate varied mock response with language detection"""
        import random
        import re

        # Simulate processing time
        await asyncio.sleep(0.1)

        # Detect language (Chinese vs English)
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', prompt))
        lang_suffix = "_zh" if has_chinese else "_en"

        # Detect intent from prompt
        prompt_lower = prompt.lower()

        # Choose response category (Chinese patterns)
        if any(word in prompt for word in ["你好", "您好", "嗨", "hi", "hello", "hey"]):
            category = "greetings"
        elif any(word in prompt for word in ["吗", "什么", "怎么", "为什么", "哪里", "谁", "?", "what", "how", "why"]):
            category = "questions"
        elif any(word in prompt for word in ["好", "棒", "开心", "高兴", "兴奋", "喜欢", "good", "great", "happy", "love"]):
            category = "positive"
        elif any(word in prompt for word in ["难过", "伤心", "不好", "糟糕", "讨厌", "生气", "sad", "bad", "angry"]):
            category = "negative"
        elif any(word in prompt for word in ["谢谢", "感谢", "thank"]):
            category = "thanks"
        else:
            category = "default"

        # Get response list with language suffix
        full_category = category + lang_suffix
        responses = self.responses.get(full_category, self.responses.get("default" + lang_suffix, self.responses.get("default_en", [])))

        if not responses:
            # Fallback to English default if category not found
            responses = self.responses.get("default_en", [
                {"utterance": "I see! Tell me more!", "emote": {"type": "neutral", "intensity": 0.5}, "intent": "ASK"}
            ])

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
            # Try Anthropic Claude
            if self.config.backend == "anthropic":
                anthropic_backend = AnthropicBackend(self.config)
                if await anthropic_backend.is_available():
                    self.backend = anthropic_backend
                    print(f"[OK] Anthropic Claude backend initialized ({self.config.model})")
                    self.is_ready = True
                    return
                else:
                    print("[WARN] Anthropic API not available, falling back to Mock")

            # Try OpenAI
            elif self.config.backend == "openai":
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
