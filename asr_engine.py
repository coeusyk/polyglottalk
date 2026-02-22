"""
asr_engine.py — Speech-to-text thread using faster-whisper.

Key constraints:
- WhisperModel is loaded ONCE in __init__ (not in run()).
- model.transcribe() returns a GENERATOR — it must be fully drained.
- Silent/hallucinated chunks are filtered before pushing to text_queue.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

import numpy as np

import config
from models import AudioChunk, TextSegment

# Import after config.py has set OMP_NUM_THREADS / CT2_INTER_THREADS
from faster_whisper import WhisperModel  # noqa: E402

logger = logging.getLogger(__name__)


class ASREngine:
    """Transcribes AudioChunk objects into TextSegment objects.

    Filters
    -------
    1. RMS silence filter — chunks whose RMS energy is below
       ``config.RMS_SILENCE_THRESHOLD`` are skipped.
    2. Duplicate filter — if the transcription is identical to the
       previous non-empty result, it is skipped (Whisper hallucination).
    """

    def __init__(
        self,
        audio_queue: queue.Queue,
        text_queue: queue.Queue,
        stop_event: threading.Event,
        model_size: str = config.ASR_MODEL_SIZE,
        compute_type: str = config.ASR_COMPUTE_TYPE,
        beam_size: int = config.ASR_BEAM_SIZE,
    ) -> None:
        self._audio_queue = audio_queue
        self._text_queue = text_queue
        self._stop_event = stop_event
        self._beam_size = beam_size

        logger.info("Loading ASR model (%s, %s)…", model_size, compute_type)
        t0 = time.perf_counter()
        self.model = WhisperModel(
            model_size,
            device=config.ASR_DEVICE,
            compute_type=compute_type,
        )
        logger.info("ASR model loaded in %.1fs", time.perf_counter() - t0)

        self._last_text: str = ""

    # ── Thread target ───────────────────────────────────────────────────────

    def run(self) -> None:
        """Consume AudioChunks, transcribe, push TextSegments."""
        while not self._stop_event.is_set():
            try:
                item = self._audio_queue.get(timeout=config.QUEUE_GET_TIMEOUT)
            except queue.Empty:
                continue

            if item is None:  # shutdown sentinel
                break

            assert isinstance(item, AudioChunk)

            # ── Silence filter ─────────────────────────────────────────────
            rms = float(np.sqrt(np.mean(item.audio ** 2)))
            if rms < config.RMS_SILENCE_THRESHOLD:
                logger.debug(
                    "Chunk #%d skipped — silent (RMS=%.4f)", item.chunk_id, rms
                )
                continue

            # ── Transcribe ─────────────────────────────────────────────────
            t0 = time.perf_counter()
            text = self._transcribe(item.audio)
            elapsed = time.perf_counter() - t0

            if not text:
                logger.debug("Chunk #%d produced empty transcript.", item.chunk_id)
                continue

            # ── Duplicate / hallucination filter ───────────────────────────
            if text == self._last_text:
                logger.debug(
                    "Chunk #%d skipped — duplicate transcript: %r",
                    item.chunk_id,
                    text,
                )
                continue

            self._last_text = text
            logger.info(
                "Transcription done (%.3fs) chunk #%d: %r",
                elapsed,
                item.chunk_id,
                text,
            )

            segment = TextSegment(
                chunk_id=item.chunk_id,
                text=text,
                timestamp=time.perf_counter(),
            )
            self._put(segment)

        logger.info("ASREngine stopped.")

    # ── Internal ────────────────────────────────────────────────────────────

    def _transcribe(self, audio: np.ndarray) -> str:
        """Run faster-whisper on a float32 16 kHz numpy array.

        The generator returned by model.transcribe() MUST be fully consumed
        before the next call — partial iteration can corrupt CTranslate2 state.
        """
        segments_gen, _info = self.model.transcribe(
            audio,
            beam_size=self._beam_size,
            language=config.ASR_LANGUAGE,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        # Drain the generator completely
        text = "".join(seg.text for seg in segments_gen).strip()
        return text

    def _put(self, segment: TextSegment) -> None:
        """Push to text_queue; retry (with stop_event check) on Full."""
        while not self._stop_event.is_set():
            try:
                self._text_queue.put(segment, timeout=config.QUEUE_PUT_TIMEOUT)
                return
            except queue.Full:
                logger.debug("text_queue full — retrying put for chunk #%d", segment.chunk_id)
