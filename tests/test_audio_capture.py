"""
tests/test_audio_capture.py — Verify microphone recording.

Requires a live microphone attached to the machine.

Run:
    python -m tests.test_audio_capture
"""

from __future__ import annotations

import queue
import threading
import time
import wave
import sys
import os

# Ensure project root is on sys.path when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import config  # sets os.environ first
import numpy as np
from audio_capture import AudioCapture
from models import AudioChunk

RECORD_SECONDS = 5          # Record this many seconds
OUTPUT_WAV = "tests/test_output.wav"
RMS_MIN = 1e-6  # Just verify audio isn't pure zeros — silent rooms have background noise ~1e-5

pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Live microphone test — skip in CI",
)


def test_audio_capture_live() -> None:
    """Record 3 s from mic, save to WAV, assert non-silence."""
    audio_queue: queue.Queue = queue.Queue(maxsize=10)
    stop_event = threading.Event()

    capture = AudioCapture(audio_queue=audio_queue, stop_event=stop_event)
    t = threading.Thread(target=capture.run, daemon=True)
    t.start()

    print(f"Recording for {RECORD_SECONDS} seconds — please speak or make noise…")
    time.sleep(RECORD_SECONDS + 0.5)
    stop_event.set()
    t.join(timeout=3)

    chunks: list[np.ndarray] = []
    while not audio_queue.empty():
        item = audio_queue.get_nowait()
        if isinstance(item, AudioChunk):
            chunks.append(item.audio)

    assert len(chunks) > 0, "No AudioChunks received — check microphone."

    audio = np.concatenate(chunks)

    # ── Save to WAV ────────────────────────────────────────────────────────
    os.makedirs("tests", exist_ok=True)
    pcm16 = (audio * 32767).astype(np.int16)
    with wave.open(OUTPUT_WAV, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(config.SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())
    print(f"Saved {len(audio) / config.SAMPLE_RATE:.1f}s of audio to {OUTPUT_WAV}")

    # ── Assertions ─────────────────────────────────────────────────────────
    file_size = os.path.getsize(OUTPUT_WAV)
    assert file_size > 1000, f"WAV file too small: {file_size} bytes"

    rms = float(np.sqrt(np.mean(audio ** 2)))
    print(f"Audio RMS: {rms:.6f}")
    assert rms > RMS_MIN, f"Audio appears completely silent (RMS={rms:.6f}) — is the microphone connected?"

    print("✓ test_audio_capture_live passed")


if __name__ == "__main__":
    test_audio_capture_live()
