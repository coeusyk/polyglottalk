"""
tts_engine.py — Text-to-speech thread using AI4Bharat IndicF5.

  Each translated segment is synthesised with IndicF5 (a high-quality
  zero-shot TTS model for Indian languages) and saved as a WAV file under
  config.TTS_OUTPUT_DIR (default: output/).

  Files are named:  output/chunk_<id>.wav
  Sample rate:      24 000 Hz (IndicF5 native output)

  The IndicF5 model is loaded lazily inside run() on the dedicated TTSThread.
  All subsequent synthesis calls reuse the loaded model.

  Voice prosody is cloned from a short reference audio clip
  (config.INDICF5_REF_AUDIO_PATH).  If config.INDICF5_REF_TEXT is left empty,
  IndicF5 auto-transcribes the reference audio with Whisper on first use;
  the result is internally cached so later calls are fast.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf

import config
from models import AudioChunk, TranslatedSegment  # noqa: F401 — for type hints in tests

logger = logging.getLogger(__name__)


class TTSEngine:
    """Synthesises TranslatedSegment text via IndicF5 and saves to WAV files.

    Each translated chunk is saved as output/chunk_<id>.wav rather than
    played through speakers, preventing microphone feedback during live
    translation sessions.

    The IndicF5 AutoModel is loaded inside run() (on the TTSThread) to keep
    __init__ cheap so the Pipeline constructor stays fast.
    """

    #: Native IndicF5 output sample rate (Hz)
    _SAMPLERATE: int = 24_000

    def __init__(
        self,
        tts_queue: "queue.Queue[Optional[TranslatedSegment]]",
        stop_event: threading.Event,
        output_dir: str = config.TTS_OUTPUT_DIR,
    ) -> None:
        self._tts_queue = tts_queue
        self._stop_event = stop_event
        self._output_dir = Path(output_dir)

        # Resolve device at construction time (no torch import yet)
        self._device: str = config.INDICF5_DEVICE

        # Model loaded in run() — do NOT load here
        self._model: Optional[Any] = None

    # ── Thread target ──────────────────────────────────────────────────────

    def run(self) -> None:
        """Load IndicF5, then synthesise translated segments into WAV files."""
        # ── Resolve "auto" device ──────────────────────────────────────────
        if self._device == "auto":
            try:
                import torch  # noqa: PLC0415
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"

        logger.info(
            "Loading IndicF5 model '%s' on device=%s…",
            config.INDICF5_MODEL_ID,
            self._device,
        )

        from transformers import AutoModel  # noqa: PLC0415

        self._model = AutoModel.from_pretrained(
            config.INDICF5_MODEL_ID,
            trust_remote_code=True,
            low_cpu_mem_usage=False,  # vocos vocoder calls .item() during init; meta tensors break this
        )

        # Move to GPU if requested
        if self._device == "cuda":
            try:
                import torch  # noqa: PLC0415
                self._model = self._model.to(torch.device("cuda"))
            except Exception:
                logger.warning("Failed to move IndicF5 to CUDA — falling back to CPU.")
                self._device = "cpu"

        self._output_dir.mkdir(parents=True, exist_ok=True)

        ref_audio = str(config.INDICF5_REF_AUDIO_PATH)
        ref_text = config.INDICF5_REF_TEXT

        # Validate reference audio exists
        if not Path(ref_audio).exists():
            logger.warning(
                "IndicF5 reference audio not found at '%s'. "
                "Run 'python setup_models.py' to download it.",
                ref_audio,
            )

        logger.info(
            "TTS engine ready (IndicF5, device=%s, ref_audio=%s)",
            self._device,
            ref_audio,
        )

        while not self._stop_event.is_set():
            try:
                item = self._tts_queue.get(timeout=config.QUEUE_GET_TIMEOUT)
            except queue.Empty:
                continue

            if item is None:  # shutdown sentinel
                break

            assert isinstance(item, TranslatedSegment)

            out_path = self._output_dir / f"chunk_{item.chunk_id:04d}.wav"

            t0 = time.perf_counter()
            self._synthesise(item.text, out_path, ref_audio, ref_text)
            elapsed = time.perf_counter() - t0

            # End-to-end latency measured from audio capture to file write
            e2e = time.perf_counter() - item.capture_timestamp
            print(f"[TTS   #{item.chunk_id:>4d}] saved → {out_path}", flush=True)
            logger.debug(
                "TTS saved (%.3fs synthesis, %.3fs e2e) chunk #%d → %s",
                elapsed,
                e2e,
                item.chunk_id,
                out_path,
            )

        logger.info("TTSEngine stopped.")

    # ── Helpers ────────────────────────────────────────────────────────────

    def _synthesise(
        self,
        text: str,
        path: Path,
        ref_audio: str,
        ref_text: str,
    ) -> None:
        """Synthesise ``text`` via IndicF5 and write a WAV file to ``path``.

        Args:
            text:      Hindi (or target-language) text to synthesise.
            path:      Destination WAV file path.
            ref_audio: Path to the reference audio clip.
            ref_text:  Transcript of the reference clip (empty → auto-transcribe).
        """
        if self._model is None:
            logger.warning("TTS model not loaded — skipping synthesis for %s.", path.name)
            return

        try:
            audio = self._model(
                text,
                ref_audio_path=ref_audio,
                ref_text=ref_text,
            )

            # IndicF5 may return int16; normalise to float32 in [-1, 1]
            if isinstance(audio, np.ndarray) and audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32_768.0

            sf.write(str(path), np.array(audio, dtype=np.float32), samplerate=self._SAMPLERATE)

        except Exception:
            logger.exception("IndicF5 synthesis failed for chunk '%s'.", path.name)
