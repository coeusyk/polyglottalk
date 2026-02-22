"""
pipeline.py — Orchestrates the four-thread PolyglotTalk pipeline.

Thread layout (all daemon threads):

    AudioCaptureThread → [audio_queue] → ASRThread
                                            → [text_queue] → TranslatorThread
                                                                → [tts_queue] → TTSThread

All model loading happens in __init__ (main thread) so models are fully
in memory before the first audio chunk arrives.

Shutdown sequence (via stop()):
    1. stop_event.set()
    2. Push None sentinel into each queue (unblocks any waiting get())
    3. Join threads in reverse order with timeout=5 s each
"""

from __future__ import annotations

import logging
import queue
import threading
import time

import config
from audio_capture import AudioCapture
from asr_engine import ASREngine
from translator import Translator
from tts_engine import TTSEngine

logger = logging.getLogger(__name__)


class Pipeline:
    """Builds, starts, and cleanly shuts down the full S2ST pipeline."""

    def __init__(self, source_lang: str = config.SOURCE_LANG, target_lang: str = config.TARGET_LANG) -> None:
        logger.info("Creating pipeline: %s → %s", source_lang, target_lang)

        # ── Shared synchronisation ────────────────────────────────────────
        self._stop_event = threading.Event()

        # ── Inter-stage queues ────────────────────────────────────────────
        self.audio_queue: queue.Queue = queue.Queue(maxsize=config.QUEUE_MAXSIZE)
        self.text_queue: queue.Queue = queue.Queue(maxsize=config.QUEUE_MAXSIZE)
        self.tts_queue: queue.Queue = queue.Queue(maxsize=config.QUEUE_MAXSIZE)

        # ── Worker instances (models loaded here, in main thread) ─────────
        # NOTE: ASREngine and Translator each load their model in __init__.
        logger.info("Loading ASR model…")
        self._asr_engine = ASREngine(
            audio_queue=self.audio_queue,
            text_queue=self.text_queue,
            stop_event=self._stop_event,
        )

        logger.info("Loading translation model (%s → %s)…", source_lang, target_lang)
        self._translator = Translator(
            text_queue=self.text_queue,
            tts_queue=self.tts_queue,
            stop_event=self._stop_event,
            source_lang=source_lang,
            target_lang=target_lang,
        )

        # AudioCapture and TTSEngine have zero-cost __init__
        self._audio_capture = AudioCapture(
            audio_queue=self.audio_queue,
            stop_event=self._stop_event,
        )
        self._tts_engine = TTSEngine(
            tts_queue=self.tts_queue,
            stop_event=self._stop_event,
        )

        # Thread handles (created in start())
        self._threads: list[threading.Thread] = []

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Create and start all four daemon threads."""
        specs = [
            ("AudioCaptureThread", self._audio_capture.run),
            ("ASRThread", self._asr_engine.run),
            ("TranslatorThread", self._translator.run),
            ("TTSThread", self._tts_engine.run),
        ]
        for name, target in specs:
            t = threading.Thread(target=target, name=name, daemon=True)
            self._threads.append(t)
            t.start()
            logger.info("Started %s", name)

        print("✓ Pipeline ready. Speak now… (Ctrl+C to stop)")

    def stop(self) -> None:
        """Signal all threads to exit and wait for them to finish."""
        logger.info("Stopping pipeline…")
        self._stop_event.set()

        # Push one sentinel per queue to unblock any thread stuck in get()
        for q in (self.audio_queue, self.text_queue, self.tts_queue):
            try:
                q.put_nowait(None)
            except queue.Full:
                # Drain one item to make room, then insert sentinel
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(None)
                except queue.Full:
                    pass

        # Join in reverse order (downstream first so upstream can drain)
        for t in reversed(self._threads):
            t.join(timeout=5)
            if t.is_alive():
                logger.warning("Thread %s did not stop in time.", t.name)

        print("Pipeline stopped.")

    def wait(self) -> None:
        """Block the calling thread until KeyboardInterrupt, then stop."""
        try:
            # Spin on a sleep so the main thread stays interruptible
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\nInterrupt received — shutting down…")
        finally:
            self.stop()
