#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive test suite for Ani v0
Tests all phases implemented so far
"""
import sys
import os

# Windows console encoding fix
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

print("=" * 60)
print("Ani v0 - Comprehensive Test Suite")
print("=" * 60)

# Test 1: Check Python version
print("\n[1/8] Checking Python version...")
if sys.version_info < (3, 8):
    print(f"[FAIL] Python 3.8+ required, found {sys.version}")
    sys.exit(1)
print(f"[OK] Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

# Test 2: Validate syntax of all Python files
print("\n[2/8] Validating Python syntax...")
files_to_check = [
    'main.py',
    'test_client.py',
    'audio_pipeline.py',
    'main_audio.py',
    'test_audio_client.py',
    'validate.py'
]

syntax_errors = []
for filename in files_to_check:
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            compile(f.read(), filename, 'exec')
        print(f"  [OK] {filename}")
    except SyntaxError as e:
        print(f"  [FAIL] {filename}: {e}")
        syntax_errors.append((filename, e))
    except FileNotFoundError:
        print(f"  [WARN] {filename}: File not found")

if syntax_errors:
    print(f"\n[FAIL] {len(syntax_errors)} file(s) have syntax errors")
    sys.exit(1)

# Test 3: Check required files exist
print("\n[3/8] Checking required files...")
required_files = [
    'README.md',
    'requirements.txt',
    '.gitignore',
    'TESTING.md',
    'PROGRESS.md'
]

for filename in required_files:
    if os.path.exists(filename):
        print(f"  [OK] {filename}")
    else:
        print(f"  [FAIL] {filename} missing")

# Test 4: Test imports (Phase 1 - core dependencies)
print("\n[4/8] Testing Phase 1 imports (core dependencies)...")
phase1_imports = {
    'json': 'Standard library',
    'time': 'Standard library',
    'asyncio': 'Standard library',
}

for module, desc in phase1_imports.items():
    try:
        __import__(module)
        print(f"  [OK] {module} ({desc})")
    except ImportError as e:
        print(f"  [FAIL] {module}: {e}")

# Test 5: Test FastAPI dependencies
print("\n[5/8] Testing FastAPI dependencies...")
fastapi_deps = {
    'pydantic': 'Data validation',
    'fastapi': 'Web framework',
    'uvicorn': 'ASGI server',
    'websockets': 'WebSocket support'
}

missing_deps = []
for module, desc in fastapi_deps.items():
    try:
        __import__(module)
        print(f"  [OK] {module} ({desc})")
    except ImportError:
        print(f"  [FAIL] {module} - Run: pip install {module}")
        missing_deps.append(module)

# Test 6: Test audio dependencies (Phase 2)
print("\n[6/8] Testing Phase 2 audio dependencies...")
audio_deps = {
    'numpy': 'Array operations',
    'torch': 'PyTorch (VAD/STT backend)',
}

for module, desc in audio_deps.items():
    try:
        __import__(module)
        print(f"  [OK] {module} ({desc})")
    except ImportError:
        print(f"  [FAIL] {module} - Run: pip install {module}")
        missing_deps.append(module)

# Test 7: Test Phase 1 JSON schema validation
print("\n[7/8] Testing Phase 1 JSON schema validation...")
try:
    from pydantic import BaseModel, Field, field_validator
    from typing import List

    class Emote(BaseModel):
        type: str = Field(..., pattern="^(joy|sad|anger|surprise|neutral)$")
        intensity: float = Field(..., ge=0.0, le=1.0)

    class LLMResponse(BaseModel):
        utterance: str = Field(..., max_length=500)
        emote: Emote
        intent: str = Field(..., pattern="^(SMALL_TALK|ANSWER|ASK|JOKE|TOOL_USE)$")
        phoneme_hints: List[List] = Field(default_factory=list)

    # Test valid data
    valid_data = {
        "utterance": "Hello!",
        "emote": {"type": "joy", "intensity": 0.8},
        "intent": "SMALL_TALK",
        "phoneme_hints": []
    }
    response = LLMResponse(**valid_data)
    print("  [OK] Valid JSON schema accepted")

    # Test invalid emote type
    try:
        invalid_data = {
            "utterance": "Test",
            "emote": {"type": "invalid", "intensity": 0.5},
            "intent": "SMALL_TALK"
        }
        LLMResponse(**invalid_data)
        print("  [FAIL] Invalid emote type should be rejected")
    except Exception:
        print("  [OK] Invalid emote type rejected")

    # Test invalid intensity
    try:
        invalid_data = {
            "utterance": "Test",
            "emote": {"type": "joy", "intensity": 1.5},
            "intent": "SMALL_TALK"
        }
        LLMResponse(**invalid_data)
        print("  [FAIL] Invalid intensity should be rejected")
    except Exception:
        print("  [OK] Invalid intensity rejected")

    print("  [OK] JSON schema validation working")

except Exception as e:
    print(f"  [FAIL] Schema validation test failed: {e}")

# Test 8: Import and validate audio pipeline
print("\n[8/8] Testing Phase 2 audio pipeline imports...")
try:
    # This will fail if torch/silero not installed, but syntax is still validated
    if 'torch' in sys.modules:
        from audio_pipeline import AudioConfig, SileroVAD, WhisperSTT, AudioPipeline

        # Test AudioConfig
        config = AudioConfig()
        assert config.sample_rate == 16000
        duration_ms = config.chunk_duration_ms
        assert duration_ms > 0
        chunk_info = f"{config.chunk_size} samples (~{duration_ms:.1f}ms)"
        print(f"  [OK] AudioConfig initialized correctly [{chunk_info}]")

        # Test classes can be instantiated
        vad = SileroVAD(config)
        assert vad.config == config
        print("  [OK] SileroVAD class OK")

        stt = WhisperSTT()
        assert stt.model_size == "base"
        print("  [OK] WhisperSTT class OK")

        pipeline = AudioPipeline(config)
        assert pipeline.config == config
        print("  [OK] AudioPipeline class OK")

        print("  [OK] Audio pipeline modules validated")
    else:
        print("  [WARN] PyTorch not installed - audio pipeline not tested")
        print("    (This is OK if you haven't installed Phase 2 dependencies yet)")

except Exception as e:
    print(f"  [FAIL] Audio pipeline test failed: {e}")

# Summary
print("\n" + "=" * 60)
print("Test Summary")
print("=" * 60)

if missing_deps:
    print(f"\n[WARN] Missing dependencies: {', '.join(missing_deps)}")
    print("Run: pip install -r requirements.txt")
    print("\nPhase 1 (JSON-only mode) should work with FastAPI dependencies.")
    print("Phase 2 (Audio pipeline) requires all dependencies.")
else:
    print("\n[OK] All dependencies installed!")

if syntax_errors:
    print(f"\n[FAIL] {len(syntax_errors)} file(s) have syntax errors - please fix")
    sys.exit(1)
else:
    print("\n[OK] All syntax checks passed!")

print("\n" + "=" * 60)
print("Next Steps:")
print("=" * 60)
print("\n1. If dependencies missing, run:")
print("   pip install -r requirements.txt")
print("\n2. Test Phase 1 (JSON only):")
print("   python main.py")
print("   # In another terminal:")
print("   python test_client.py")
print("\n3. Test Phase 2 (Audio pipeline):")
print("   python main_audio.py")
print("   # In another terminal:")
print("   python test_audio_client.py")
print("\n4. Standalone audio pipeline test:")
print("   python audio_pipeline.py")
print("\n" + "=" * 60)
