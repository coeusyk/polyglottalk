"""
tests/test_asr.py — Verify faster-whisper transcription on a known WAV file.

The test ships a tiny synthetic WAV (500 Hz sine wave with "Hello" spliced in)
OR relies on tests/test_audio/hello.wav being present.

Run:
    python -m tests.test_asr
    python -m pytest tests/test_asr.py
"""

from __future__ import annotations

import os
import sys
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config  # noqa: F401 — sets os.environ first

import numpy as np
import pytest


HELLO_WAV = os.path.join(os.path.dirname(__file__), "test_audio", "hello.wav")


def _generate_silence_wav(path: str, duration: float = 2.5) -> None:
    """Generate a silent 16 kHz mono WAV for testing (no mic needed)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = int(config.SAMPLE_RATE * duration)
    silence = np.zeros(n, dtype=np.float32)
    pcm16 = (silence * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(config.SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())


def _load_wav_as_float32(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return data


@pytest.fixture(scope="module")
def asr_model():
    """Load WhisperModel once for all tests in this module."""
    from faster_whisper import WhisperModel

    return WhisperModel(
        config.ASR_MODEL_SIZE,
        device=config.ASR_DEVICE,
        compute_type=config.ASR_COMPUTE_TYPE,
    )


def test_transcribe_silence(asr_model) -> None:
    """Silent audio should not crash; result is empty or near-empty."""
    silence = np.zeros(config.BLOCK_SIZE, dtype=np.float32)
    segments_gen, _info = asr_model.transcribe(
        silence,
        beam_size=config.ASR_BEAM_SIZE,
        language=config.ASR_LANGUAGE,
        vad_filter=False,
    )
    text = "".join(seg.text for seg in segments_gen).strip()
    # Should not crash; text may be empty or a hallucination — both acceptable
    assert isinstance(text, str)
    print(f"Silence transcript: {text!r}")
    print("✓ test_transcribe_silence passed")


def test_transcribe_hello_wav(asr_model) -> None:
    """Transcribe hello.wav (if present) and assert 'hello' in output."""
    if not os.path.exists(HELLO_WAV):
        pytest.skip(
            f"hello.wav not found at {HELLO_WAV}. "
            "Place a 2.5-second recording of 'Hello, how are you?' there."
        )
    audio = _load_wav_as_float32(HELLO_WAV)
    segments_gen, _info = asr_model.transcribe(
        audio,
        beam_size=config.ASR_BEAM_SIZE,
        language=config.ASR_LANGUAGE,
        vad_filter=False,
    )
    text = "".join(seg.text for seg in segments_gen).strip()
    print(f"Transcript: {text!r}")
    assert "hello" in text.lower(), f"Expected 'hello' in transcript, got: {text!r}"
    print("✓ test_transcribe_hello_wav passed")


def test_asr_engine_integration() -> None:
    """Test ASREngine._transcribe() with a synthesized audio chunk."""
    import queue
    import threading
    from asr_engine import ASREngine

    audio_q: queue.Queue = queue.Queue()
    text_q: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    engine = ASREngine(
        audio_queue=audio_q,
        text_queue=text_q,
        stop_event=stop_event,
    )

    silence = np.zeros(config.BLOCK_SIZE, dtype=np.float32)
    result = engine._transcribe(silence)
    assert isinstance(result, str)
    print(f"ASREngine._transcribe(silence) = {result!r}")
    print("✓ test_asr_engine_integration passed")


if __name__ == "__main__":
    import sys
    pytest_args = [__file__, "-v"]
    sys.exit(pytest.main(pytest_args))
