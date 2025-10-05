"""
TTS Pipeline for Ani v0
Text-to-Speech with phoneme extraction for lipsync
Target: <700ms latency
"""
import asyncio
import time
import io
import wave
from typing import Optional, List, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class TTSConfig:
    """TTS configuration"""
    engine: str = "edge"  # edge, pyttsx3, coqui
    voice: str = "en-US-AriaNeural"  # Edge TTS voice or Coqui model name
    rate: str = "+0%"  # Speech rate (Edge TTS only)
    pitch: str = "+0Hz"  # Pitch (Edge TTS only)
    sample_rate: int = 24000  # Output sample rate
    format: str = "riff-24khz-16bit-mono-pcm"  # Edge output format (WAV by default)
    speaker_wav: Optional[str] = None  # Path to speaker WAV file (Coqui only)


class SimplePhonemizer:
    """
    Simple phoneme estimation from text
    Maps words to approximate phoneme timing
    """

    # Basic phoneme duration estimates (ms per phoneme)
    PHONEME_DURATIONS = {
        "vowel": 80,      # a, e, i, o, u
        "consonant": 60,  # most consonants
        "silence": 100,   # pauses
    }

    VOWELS = set('aeiouAEIOU')

    @staticmethod
    def estimate_phonemes(text: str, audio_duration_ms: float) -> List[Tuple[str, float, float]]:
        """
        Estimate phoneme timing from text
        Returns: [(phoneme, start_ms, end_ms), ...]
        """
        # Clean text
        text = ''.join(c for c in text if c.isalnum() or c.isspace())
        words = text.split()

        if not words:
            return []

        phonemes = []
        current_time = 0.0

        # Estimate time per character
        total_chars = sum(len(word) for word in words)
        if total_chars == 0:
            return []

        time_per_char = audio_duration_ms / total_chars

        for word in words:
            for char in word.lower():
                if char in SimplePhonemizer.VOWELS:
                    duration = time_per_char * 1.2  # Vowels slightly longer
                    phoneme = char.upper()
                else:
                    duration = time_per_char * 0.8
                    phoneme = char.upper()

                phonemes.append((phoneme, current_time, current_time + duration))
                current_time += duration

            # Add space/pause
            phonemes.append(("_", current_time, current_time + time_per_char * 0.5))
            current_time += time_per_char * 0.5

        return phonemes


class EdgeTTSEngine:
    """
    Edge TTS engine (Microsoft Azure TTS)
    Fast, high quality, free
    """

    def __init__(self, config: TTSConfig):
        self.config = config
        self.is_ready = True

    async def synthesize(self, text: str) -> Tuple[bytes, float]:
        """
        Synthesize speech from text
        Returns: (audio_bytes, duration_ms)
        """
        try:
            import edge_tts

            # Create TTS communicator
            communicate = edge_tts.Communicate(
                text,
                self.config.voice,
                rate=self.config.rate,
                pitch=self.config.pitch
            )

            # Collect audio chunks
            audio_data = io.BytesIO()

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.write(chunk["data"])

            audio_bytes = audio_data.getvalue()
            duration_ms = self._estimate_duration(audio_bytes, text)

            return audio_bytes, duration_ms

        except Exception as e:
            print(f"[FAIL] Edge TTS error: {e}")
            raise

    def _estimate_duration(self, audio_bytes: bytes, text: str) -> float:
        """Estimate duration from WAV metadata when available
        Fallback heuristics keep audio responsive even if parsing fails
        """
        try:
            with wave.open(io.BytesIO(audio_bytes), 'rb') as wav_file:
                frames = wav_file.getnframes()
                framerate = wav_file.getframerate() or self.config.sample_rate
                if framerate > 0:
                    return (frames / framerate) * 1000
        except wave.Error:
            pass

        if self.config.sample_rate > 0:
            bytes_per_sample = 2  # 16-bit audio
            channels = 1
            bytes_per_second = self.config.sample_rate * bytes_per_sample * channels
            if bytes_per_second > 0:
                return (len(audio_bytes) / bytes_per_second) * 1000

        return max(len(text) * 50, 1)


class CoquiTTSEngine:
    """
    Coqui TTS engine with XTTS-v2
    Natural sounding, voice cloning capable
    """

    def __init__(self, config: TTSConfig):
        self.config = config
        self.tts = None
        self.is_ready = False

    def initialize(self):
        """Initialize Coqui TTS engine"""
        try:
            import torch
            from TTS.api import TTS

            device = "cuda" if torch.cuda.is_available() else "cpu"

            print(f"Loading Coqui TTS model: {self.config.voice}")
            # Auto-agree to non-commercial CPML license
            import os
            os.environ["COQUI_TOS_AGREED"] = "1"
            self.tts = TTS(self.config.voice, progress_bar=False).to(device)

            self.is_ready = True
            print(f"[OK] Coqui TTS initialized on {device}")
        except Exception as e:
            print(f"[FAIL] Coqui TTS init error: {e}")
            raise

    async def synthesize(self, text: str) -> Tuple[bytes, float]:
        """
        Synthesize speech from text
        Returns: (audio_bytes, duration_ms)
        """
        if not self.is_ready:
            self.initialize()

        try:
            import tempfile
            import os
            import wave
            import time

            print(f"[Coqui] Synthesizing: '{text[:50]}...'")
            start_time = time.time()

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name

            # Auto-detect language (Chinese or English)
            def detect_language(text):
                # Simple detection: if contains Chinese characters, use zh-cn
                import re
                if re.search(r'[\u4e00-\u9fff]', text):
                    return "zh-cn"
                return "en"

            language = detect_language(text)
            print(f"[Coqui] Detected language: {language}")

            # Generate speech with or without voice cloning
            if self.config.speaker_wav and os.path.exists(self.config.speaker_wav):
                # Voice cloning mode (for XTTS-v2)
                print(f"[Coqui] Using voice cloning with {self.config.speaker_wav}")
                self.tts.tts_to_file(
                    text=text,
                    speaker_wav=self.config.speaker_wav,
                    language=language,
                    file_path=temp_path
                )
            else:
                # Default voice mode
                print(f"[Coqui] Using model built-in voice")
                # Simple TTS without speaker (works for models like Tacotron2, VITS, etc.)
                self.tts.tts_to_file(
                    text=text,
                    file_path=temp_path
                )

            # Read audio data
            with open(temp_path, 'rb') as f:
                audio_bytes = f.read()

            # Get actual duration from WAV file
            try:
                with wave.open(temp_path, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    duration_ms = (frames / float(rate)) * 1000
            except:
                # Fallback estimate
                duration_ms = len(text) * 60

            # Clean up
            os.unlink(temp_path)

            elapsed = time.time() - start_time
            print(f"[Coqui] Generated {duration_ms:.0f}ms audio in {elapsed:.2f}s")

            return audio_bytes, duration_ms

        except Exception as e:
            print(f"[FAIL] Coqui TTS synthesis error: {e}")
            import traceback
            traceback.print_exc()
            raise


class PyTTSX3Engine:
    """
    pyttsx3 engine (offline, local)
    Faster but lower quality
    """

    def __init__(self, config: TTSConfig):
        self.config = config
        self.engine = None
        self.is_ready = False

    def initialize(self):
        """Initialize pyttsx3 engine"""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)  # Speed
            self.engine.setProperty('volume', 1.0)
            self.is_ready = True
            print("[OK] pyttsx3 engine initialized")
        except Exception as e:
            print(f"[FAIL] pyttsx3 init error: {e}")
            raise

    async def synthesize(self, text: str) -> Tuple[bytes, float]:
        """
        Synthesize speech from text
        Returns: (audio_bytes, duration_ms)
        """
        if not self.is_ready:
            self.initialize()

        try:
            # Save to temp file
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name

            # Generate speech
            self.engine.save_to_file(text, temp_path)
            self.engine.runAndWait()

            # Read audio data
            with open(temp_path, 'rb') as f:
                audio_bytes = f.read()

            # Clean up
            os.unlink(temp_path)

            # Estimate duration
            duration_ms = len(text) * 60

            return audio_bytes, duration_ms

        except Exception as e:
            print(f"[FAIL] pyttsx3 synthesis error: {e}")
            raise


class TTSPipeline:
    """
    Complete TTS pipeline with phoneme extraction
    """

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self.engine = None
        self.phonemizer = SimplePhonemizer()
        self.is_ready = False

    async def initialize(self):
        """Initialize TTS engine"""
        print(f"Initializing TTS pipeline (engine: {self.config.engine})...")

        try:
            if self.config.engine == "edge":
                self.engine = EdgeTTSEngine(self.config)
                print(f"[OK] Edge TTS initialized (voice: {self.config.voice})")
                self.is_ready = True
            elif self.config.engine == "coqui":
                self.engine = CoquiTTSEngine(self.config)
                self.engine.initialize()
                self.is_ready = True
            elif self.config.engine == "pyttsx3":
                self.engine = PyTTSX3Engine(self.config)
                self.engine.initialize()
                self.is_ready = True
            else:
                raise ValueError(f"Unknown TTS engine: {self.config.engine}")

        except Exception as e:
            print(f"[FAIL] TTS initialization failed: {e}")
            # Fallback to edge TTS (most reliable)
            try:
                print("[WARN] Falling back to Edge TTS")
                self.config.engine = "edge"
                self.engine = EdgeTTSEngine(self.config)
                self.is_ready = True
                print("[OK] Edge TTS fallback initialized")
            except:
                raise

    async def synthesize_with_phonemes(self, text: str) -> dict:
        """
        Synthesize speech and extract phonemes

        Returns:
            {
                "audio": bytes,
                "duration_ms": float,
                "phonemes": [(phoneme, start_ms, end_ms), ...],
                "tts_latency_ms": float
            }
        """
        if not self.is_ready:
            raise RuntimeError("TTS pipeline not initialized")

        start_time = time.time()

        try:
            # Synthesize audio
            audio_bytes, duration_ms = await self.engine.synthesize(text)

            # Estimate phonemes
            phonemes = self.phonemizer.estimate_phonemes(text, duration_ms)

            # Calculate latency
            tts_latency_ms = (time.time() - start_time) * 1000

            return {
                "audio": audio_bytes,
                "duration_ms": duration_ms,
                "phonemes": phonemes,
                "tts_latency_ms": tts_latency_ms,
                "sample_rate": self.config.sample_rate
            }

        except Exception as e:
            print(f"[FAIL] TTS synthesis error: {e}")
            raise


# Testing
async def test_tts_pipeline():
    """Test TTS pipeline"""
    print("=" * 60)
    print("Testing TTS Pipeline")
    print("=" * 60)

    config = TTSConfig(engine="edge", voice="en-US-AriaNeural")
    pipeline = TTSPipeline(config)
    await pipeline.initialize()

    test_phrases = [
        "Hello! I'm Ani!",
        "How are you today?",
        "That's wonderful to hear!",
        "I'm here to help you with anything you need.",
    ]

    print("\nTest Phrases:")
    for i, text in enumerate(test_phrases, 1):
        print(f"\n[Test {i}] Text: {text}")

        result = await pipeline.synthesize_with_phonemes(text)

        print(f"[Duration] {result['duration_ms']:.0f}ms")
        print(f"[Phonemes] {len(result['phonemes'])} phonemes")
        print(f"[TTS Latency] {result['tts_latency_ms']:.0f}ms")
        print(f"[Audio Size] {len(result['audio'])} bytes")

        if result['tts_latency_ms'] < 700:
            print("[OK] Latency within target (<700ms)")
        else:
            print("[WARN] Latency exceeds target")

        # Show first few phonemes
        if result['phonemes']:
            print(f"[Sample Phonemes] {result['phonemes'][:5]}...")

    print("\n" + "=" * 60)
    print("TTS Pipeline Tests Complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_tts_pipeline())
