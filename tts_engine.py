"""
tts_engine.py — Text-to-speech thread using pyttsx3.

  Windows: uses SAPI5 (COM) backend.
  Linux:   uses espeak-ng backend (`sudo apt install espeak-ng`).

CRITICAL: pyttsx3.init() is called as the FIRST statement in run(), not
in __init__.  On Windows SAPI5 the COM apartment must belong to the
thread that calls runAndWait().  Keeping init inside run() is equally
correct on Linux and makes the class cross-platform.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

import pyttsx3

import config
from models import AudioChunk, TranslatedSegment  # noqa: F401 — for type hints in tests

logger = logging.getLogger(__name__)


class TTSEngine:
    """Speaks TranslatedSegment text aloud using pyttsx3 / SAPI5.

    pyttsx3 is NOT imported or initialised until run() is executing on
    the dedicated TTSThread.
    """

    def __init__(
        self,
        tts_queue: queue.Queue,
        stop_event: threading.Event,
        rate: int = config.TTS_RATE,
    ) -> None:
        self._tts_queue = tts_queue
        self._stop_event = stop_event
        self._rate = rate

        # Engine is created in run() — do NOT init here
        self._engine: Optional[pyttsx3.Engine] = None

    # ── Thread target ───────────────────────────────────────────────────────

    def run(self) -> None:
        """Initialise SAPI5 engine (COM), speak queued translations."""
        # ── COM init must be FIRST inside this thread ──────────────────────
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", self._rate)

        self._select_voice()
        logger.info("TTS engine initialized (rate=%d wpm)", self._rate)

        while not self._stop_event.is_set():
            try:
                item = self._tts_queue.get(timeout=config.QUEUE_GET_TIMEOUT)
            except queue.Empty:
                continue

            if item is None:  # shutdown sentinel
                break

            assert isinstance(item, TranslatedSegment)

            t0 = time.perf_counter()
            self._speak(item.text)
            elapsed = time.perf_counter() - t0

            # End-to-end latency: from audio capture to speech completion
            e2e = time.perf_counter() - item.timestamp
            logger.info(
                "Speech done (%.3fs tts, %.3fs e2e) chunk #%d: %r",
                elapsed,
                e2e,
                item.chunk_id,
                item.text,
            )

        if self._engine is not None:
            self._engine.stop()

        logger.info("TTSEngine stopped.")

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _speak(self, text: str) -> None:
        """Synthesise and play ``text`` synchronously."""
        if self._engine is not None:
            self._engine.say(text)
            self._engine.runAndWait()

    def _select_voice(self) -> None:
        """Attempt to select a Hindi voice if one is available.

        Falls back silently to the default system voice — the pipeline
        still works (with the default voice speaking transliterated Hindi).

        Voice ID formats differ by platform:
          Windows SAPI5: contains 'hi-in' or 'hindi' (case-insensitive)
          Linux espeak-ng: path ends in '/hi' or name contains 'hindi'
        """
        if self._engine is None:
            return

        voices = self._engine.getProperty("voices")
        hi_voice = next(
            (
                v for v in voices
                if "hindi" in v.name.lower()
                or "hi-in" in v.id.lower()
                or v.id.lower().endswith("/hi")
                or v.id.lower() == "hi"
            ),
            None,
        )
        if hi_voice:
            self._engine.setProperty("voice", hi_voice.id)
            logger.info("Selected Hindi voice: %s (id=%s)", hi_voice.name, hi_voice.id)
        else:
            logger.warning(
                "No Hindi TTS voice found — using default voice. "
                "Linux: install espeak-ng and an mbrola Hindi voice "
                "(`sudo apt install espeak-ng mbrola-hi1` if available). "
                "Windows: install the Hindi language pack from Settings → "
                "Time & Language → Language & Region."
            )
