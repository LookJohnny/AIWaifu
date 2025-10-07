"""
Audio Pipeline for Ani v0
Handles VAD (Voice Activity Detection) and STT (Speech-to-Text)
Target: VAD <150ms, STT <300ms
"""
import asyncio
import time
import numpy as np
from typing import Optional, AsyncGenerator, Callable
from dataclasses import dataclass
import torch


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 16000  # 16kHz for Whisper and Silero VAD
    chunk_size: int = 512  # Fixed for Silero VAD (512 samples at 16kHz)
    vad_threshold: float = 0.45  # Voice probability threshold (lower = more sensitive)
    min_speech_duration_ms: int = 300  # Minimum speech duration
    min_silence_duration_ms: int = 700  # Silence before speech ends (longer = less cutting)

    @property
    def chunk_duration_ms(self) -> float:
        """Duration per chunk in milliseconds"""
        return (self.chunk_size / self.sample_rate) * 1000


class SileroVAD:
    """
    Silero VAD wrapper
    Target latency: <150ms
    """
    def __init__(self, config: AudioConfig):
        self.config = config
        self.model = None
        self.is_loaded = False

    async def load(self):
        """Load Silero VAD model"""
        if self.is_loaded:
            return

        print("Loading Silero VAD model...")
        start = time.time()

        try:
            # Load Silero VAD from torch hub
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )

            # Extract utility functions
            (get_speech_timestamps, _, read_audio, *_) = utils
            self.get_speech_timestamps = get_speech_timestamps

            self.is_loaded = True
            elapsed = (time.time() - start) * 1000
            print(f"[OK] Silero VAD loaded in {elapsed:.0f}ms")

        except Exception as e:
            print(f"[FAIL] Failed to load Silero VAD: {e}")
            raise

    def detect_speech(self, audio_chunk: np.ndarray) -> float:
        """
        Detect speech in audio chunk
        Returns: speech probability (0.0-1.0)
        """
        if not self.is_loaded:
            raise RuntimeError("VAD model not loaded")

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_chunk).float()

        # Get speech probability
        with torch.no_grad():
            speech_prob = self.model(audio_tensor, self.config.sample_rate).item()

        return speech_prob

    def is_speech(self, speech_prob: float) -> bool:
        """Check if probability exceeds threshold"""
        return speech_prob >= self.config.vad_threshold


class WhisperSTT:
    """
    Faster-Whisper STT wrapper
    Target latency: <300ms for streaming partials
    """
    def __init__(self, model_size: str = "base", device: str = "auto"):
        self.model_size = model_size
        self.device = device
        self.model = None
        self.is_loaded = False

    async def load(self):
        """Load Faster-Whisper model"""
        if self.is_loaded:
            return

        print(f"Loading Faster-Whisper ({self.model_size}) model...")
        start = time.time()

        try:
            from faster_whisper import WhisperModel

            # Auto-detect device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load model
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type="float32" if self.device == "cpu" else "float16"
            )

            self.is_loaded = True
            elapsed = (time.time() - start) * 1000
            print(f"[OK] Faster-Whisper loaded in {elapsed:.0f}ms (device: {self.device})")

        except Exception as e:
            print(f"[FAIL] Failed to load Faster-Whisper: {e}")
            raise

    async def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> str:
        """
        Transcribe audio to text
        Returns: transcribed text
        """
        if not self.is_loaded:
            raise RuntimeError("STT model not loaded")

        try:
            # Transcribe
            if language is None:
                # Let faster-whisper auto-detect language
                segments, info = self.model.transcribe(
                    audio,
                    language=None,
                    task="transcribe",
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    vad_filter=False,
                )
                try:
                    print(f"[STT] Detected language: {getattr(info, 'language', 'auto')}")
                except Exception:
                    pass
            else:
                # Use initial_prompt to improve Chinese recognition
                initial_prompt = "以下是普通话的句子。" if language == "zh" else None
                segments, info = self.model.transcribe(
                    audio,
                    language=language,
                    task="transcribe",
                    beam_size=5,  # Better beam search for accuracy
                    best_of=5,
                    temperature=0.0,
                    vad_filter=False,  # We handle VAD separately
                    initial_prompt=initial_prompt
                )

            # Collect segments
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text)

            return " ".join(text_parts).strip()

        except Exception as e:
            print(f"[FAIL] STT error: {e}")
            return ""


class AudioPipeline:
    """
    Complete audio input pipeline: VAD → STT
    Manages streaming audio and speech detection
    """
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.vad = SileroVAD(self.config)
        self.stt = WhisperSTT(model_size="small")  # Better accuracy for Chinese

        # State tracking
        self.is_speaking = False
        self.speech_buffer = []
        self.silence_duration = 0
        self.speech_duration = 0

    async def load_models(self):
        """Load VAD and STT models"""
        await self.vad.load()
        await self.stt.load()
        print("[OK] Audio pipeline ready")

    async def process_audio_stream(
        self,
        audio_generator: AsyncGenerator[bytes, None],
        on_partial: Optional[Callable[[str], None]] = None,
        on_final: Optional[Callable[[str], None]] = None
    ):
        """
        Process streaming audio with VAD and STT

        Args:
            audio_generator: Async generator yielding audio bytes
            on_partial: Callback for partial transcriptions
            on_final: Callback for final transcription
        """
        try:
            async for audio_bytes in audio_generator:
                # Convert bytes to numpy array
                audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                # Measure VAD latency
                vad_start = time.time()
                speech_prob = self.vad.detect_speech(audio_chunk)
                vad_latency = (time.time() - vad_start) * 1000

                is_speech = self.vad.is_speech(speech_prob)

                if is_speech:
                    # Speech detected
                    if not self.is_speaking:
                        self.is_speaking = True
                        self.speech_buffer = []
                        print(f"[SPEECH] Started (VAD: {vad_latency:.1f}ms, prob: {speech_prob:.2f})")

                    self.speech_buffer.append(audio_chunk)
                    self.speech_duration += self.config.chunk_duration_ms
                    self.silence_duration = 0

                else:
                    # Silence detected
                    if self.is_speaking:
                        self.silence_duration += self.config.chunk_duration_ms

                        # Check if speech ended
                        if self.silence_duration >= self.config.min_silence_duration_ms:
                            # Speech ended - transcribe
                            if self.speech_duration >= self.config.min_speech_duration_ms:
                                audio_array = np.concatenate(self.speech_buffer)

                                print(f"[SPEECH] Ended ({self.speech_duration}ms)")

                                # Measure STT latency
                                stt_start = time.time()
                                text = await self.stt.transcribe(audio_array, language="zh")
                                stt_latency = (time.time() - stt_start) * 1000

                                # Safe print with encoding error handling
                                try:
                                    print(f"[STT] Transcribed in {stt_latency:.0f}ms: '{text}'")
                                except (UnicodeEncodeError, UnicodeDecodeError):
                                    print(f"[STT] Transcribed in {stt_latency:.0f}ms: [text with encoding issues]")

                                # Call final callback
                                if on_final and text:
                                    on_final(text)

                            # Reset state
                            self.is_speaking = False
                            self.speech_buffer = []
                            self.speech_duration = 0
                            self.silence_duration = 0

        except Exception as e:
            print(f"[FAIL] Audio pipeline error: {e}")
            raise


# Testing utilities
async def test_pipeline():
    """Test the audio pipeline with synthetic audio"""
    print("Testing Audio Pipeline...")

    config = AudioConfig()
    pipeline = AudioPipeline(config)

    # Load models
    await pipeline.load_models()

    # Generate test audio (1 second of sine wave)
    duration = 1.0
    sample_rate = config.sample_rate
    frequency = 440  # A4 note

    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = (np.sin(2 * np.pi * frequency * t) * 0.3).astype(np.float32)

    # Convert to int16 bytes
    audio_bytes = (audio * 32768).astype(np.int16).tobytes()

    # Create async generator
    async def audio_generator():
        chunk_size = config.chunk_size * 2  # 2 bytes per sample
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            yield chunk
            await asyncio.sleep(config.chunk_duration_ms / 1000)

    # Process
    def on_final(text):
        print(f"\n[OK] Final transcription: {text}")

    await pipeline.process_audio_stream(audio_generator(), on_final=on_final)

    print("\n[OK] Test complete")


if __name__ == "__main__":
    asyncio.run(test_pipeline())
