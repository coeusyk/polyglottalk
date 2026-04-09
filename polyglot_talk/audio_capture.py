"""
audio_capture.py — Microphone input thread for PolyglotTalk.

Opens a sounddevice InputStream and assembles raw callback blocks into
fixed-size AudioChunk objects, pushing them onto audio_queue.

Overlapping chunks
------------------
Consecutive chunks share ``config.CHUNK_OVERLAP`` seconds of audio so that
words straddling a chunk boundary are never cut mid-phoneme.  The effective
stride (new audio per chunk) is ``CHUNK_DURATION − CHUNK_OVERLAP``.

    chunk 0:  [0 ────────── BLOCK_SIZE]
    chunk 1:      [STRIDE ────────── STRIDE+BLOCK_SIZE]
    chunk 2:          [2·STRIDE ──── 2·STRIDE+BLOCK_SIZE]

Research basis  (see config.py for full citations):
  • Whispy  — shifting buffer with re-transcription + Levenshtein dedup
  • Whisper-Streaming — LocalAgreement-2 over overlapping re-transcriptions
  • Whisper long-form — overlapping 30 s windows with timestamp stitching

Backpressure: if audio_queue is full the current chunk is DROPPED and a
warning is logged — we never accumulate unbounded audio memory.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

import numpy as np
import sounddevice as sd

from . import config
from .models import AudioChunk

logger = logging.getLogger(__name__)


class AudioCapture:
    """Captures microphone audio in overlapping chunks.

    Architecture
    ------------
    sounddevice callback (internal SD thread)
        → self._raw_q  (thread-safe Queue of raw 1-D float32 blocks)
            → run() assembles blocks into BLOCK_SIZE chunks with CHUNK_OVERLAP
                → audio_queue (shared pipeline queue)

    Each emitted chunk is BLOCK_SIZE samples long.  Consecutive chunks share
    OVERLAP_SAMPLES samples, so the stride (new audio per chunk) equals
    STRIDE_SAMPLES = BLOCK_SIZE − OVERLAP_SAMPLES.

    The double-queue pattern avoids numpy operations inside the real-time
    callback.  All assembly happens in the run() thread.
    """

    def __init__(
        self,
        audio_queue: queue.Queue,
        stop_event: threading.Event,
        sample_rate: int = config.SAMPLE_RATE,
        chunk_duration: float = config.CHUNK_DURATION,
        audio_stop_event: threading.Event | None = None,
    ) -> None:
        self._audio_queue = audio_queue
        self._stop_event = stop_event
        # Dedicated event that drain() sets to stop capture without killing
        # the downstream ASR/Translator/TTS drain.  Falls back to stop_event
        # if not supplied (e.g. in unit tests).
        self._audio_stop_event = audio_stop_event if audio_stop_event is not None else stop_event
        self._sample_rate = sample_rate
        self._block_size = int(sample_rate * chunk_duration)
        self._overlap_samples = config.OVERLAP_SAMPLES
        self._stride_samples = config.STRIDE_SAMPLES

        # Internal queue: callback → assembly loop (no lock needed)
        self._raw_q: queue.Queue[np.ndarray] = queue.Queue()

        self._chunk_id: int = 0

    # ── sounddevice callback (runs in SD internal thread) ──────────────────

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            logger.warning("sounddevice status: %s", status)
        # indata shape: (frames, channels) — take channel 0, make 1-D copy
        self._raw_q.put_nowait(indata[:, 0].copy())

    # ── Thread target ───────────────────────────────────────────────────────

    def run(self) -> None:
        """Open microphone stream, assemble overlapping chunks, push to audio_queue.

        Overlap strategy
        ----------------
        A rolling buffer accumulates raw blocks.  Whenever it holds at least
        BLOCK_SIZE samples a full chunk is emitted.  Instead of discarding
        the entire chunk (no-overlap, old behaviour), we only advance by
        STRIDE_SAMPLES, keeping the last OVERLAP_SAMPLES as the start of the
        next chunk.  This guarantees that words at boundaries appear in TWO
        consecutive chunks, giving Whisper enough acoustic context to
        transcribe them correctly on at least one side.
        """
        buffer: list[np.ndarray] = []
        buffer_samples: int = 0

        with sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        ):
            logger.info(
                "Microphone stream opened (%d Hz, mono, overlap=%dms, stride=%dms)",
                self._sample_rate,
                int(self._overlap_samples / self._sample_rate * 1000),
                int(self._stride_samples / self._sample_rate * 1000),
            )

            while not self._stop_event.is_set() and not self._audio_stop_event.is_set():
                # Drain one raw block from the callback queue
                try:
                    block = self._raw_q.get(timeout=config.QUEUE_GET_TIMEOUT)
                except queue.Empty:
                    continue

                buffer.append(block)
                buffer_samples += len(block)

                # Emit overlapping chunks as they become available
                while buffer_samples >= self._block_size:
                    full = np.concatenate(buffer)
                    chunk_audio = full[: self._block_size]

                    # Advance by stride (keep overlap for next chunk)
                    remainder = full[self._stride_samples :]

                    buffer = [remainder] if len(remainder) > 0 else []
                    buffer_samples = len(remainder)

                    item = AudioChunk(
                        chunk_id=self._chunk_id,
                        audio=chunk_audio,
                        timestamp=time.perf_counter(),
                    )
                    self._chunk_id += 1
                    self._push(item)

        logger.info("AudioCapture stopped.")

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _push(self, item: AudioChunk) -> None:
        """Push chunk to audio_queue with drop-oldest strategy on Full.

        If the queue is full the oldest unprocessed chunk is evicted so
        ASR always sees the most recent audio.  This method never blocks,
        preserving true pipeline parallelism.
        """
        try:
            self._audio_queue.put_nowait(item)
        except queue.Full:
            try:
                dropped = self._audio_queue.get_nowait()
                logger.warning(
                    "audio_queue full — evicted oldest chunk #%d to insert chunk #%d",
                    dropped.chunk_id,
                    item.chunk_id,
                )
            except queue.Empty:
                pass
            try:
                self._audio_queue.put_nowait(item)
            except queue.Full:
                logger.warning(
                    "audio_queue still full — dropping chunk #%d", item.chunk_id
                )
