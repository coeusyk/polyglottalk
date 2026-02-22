"""
tests/test_tts.py — Verify pyttsx3 SAPI5 initialisation inside a thread.

Requires speakers or headphones — the test will audibly say a phrase.

Run:
    python -m tests.test_tts
    python -m pytest tests/test_tts.py
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config  # noqa: F401

import pytest


def test_tts_speaks_in_thread() -> None:
    """pyttsx3 initialised inside child thread must speak without crashing."""
    results: dict[str, object] = {}

    def _speak() -> None:
        try:
            import pyttsx3  # noqa: PLC0415

            engine = pyttsx3.init()
            engine.setProperty("rate", config.TTS_RATE)
            engine.say("Testing one two three")
            engine.runAndWait()
            engine.stop()
            results["ok"] = True
        except Exception as exc:  # pylint: disable=broad-except
            results["error"] = str(exc)

    t = threading.Thread(target=_speak, name="TestTTSThread", daemon=True)
    t.start()
    t.join(timeout=10)

    assert not t.is_alive(), "TTS thread did not finish within 10 s"
    if "error" in results:
        pytest.fail(f"pyttsx3 raised an exception in thread: {results['error']}")

    assert results.get("ok"), "pyttsx3 did not signal success"
    print("✓ test_tts_speaks_in_thread passed")


def test_tts_engine_class() -> None:
    """TTSEngine.run() completes a speak cycle and exits on sentinel."""
    results: dict[str, object] = {}

    import queue as _queue
    from tts_engine import TTSEngine
    from models import TranslatedSegment

    tts_q: _queue.Queue = _queue.Queue()
    stop_event = threading.Event()

    engine = TTSEngine(tts_queue=tts_q, stop_event=stop_event)

    def _run() -> None:
        try:
            engine.run()
            results["ok"] = True
        except Exception as exc:  # pylint: disable=broad-except
            results["error"] = str(exc)

    t = threading.Thread(target=_run, name="TTSEngineTest", daemon=True)
    t.start()

    # Give SAPI5 a moment to initialise
    time.sleep(1.0)

    # Push one item then a sentinel
    seg = TranslatedSegment(chunk_id=0, text="Hello from the test suite")
    tts_q.put(seg)
    tts_q.put(None)  # sentinel

    t.join(timeout=15)
    assert not t.is_alive(), "TTSEngine thread did not exit after sentinel"

    if "error" in results:
        pytest.fail(f"TTSEngine.run() raised: {results['error']}")
    assert results.get("ok")
    print("✓ test_tts_engine_class passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
