"""
tts_engine.py — Text-to-speech thread using Facebook MMS-TTS (VITS-based).

  Each translated segment is synthesised with MMS-TTS (a non-autoregressive
  VITS model with constant latency regardless of text length) and saved as
  a WAV file under config.TTS_OUTPUT_DIR (default: output/).

  Files are named:  output/chunk_<id>.wav
  Sample rate:      model.config.sampling_rate (typically 16 000 Hz)

  The MMS-TTS model and tokenizer are loaded lazily inside run() on the
  dedicated TTSThread.  All subsequent synthesis calls reuse the loaded model.

  No reference audio or reference text is required — MMS-TTS is a fixed-voice
  model; the target language is encoded in config.MMS_TTS_MODEL_ID.
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
from models import TranslatedSegment  # noqa: F401 — for type hints in tests

logger = logging.getLogger(__name__)


class TTSEngine:
    """Synthesises TranslatedSegment text via MMS-TTS and saves to WAV files.

    Each translated chunk is saved as output/chunk_<id>.wav rather than
    played through speakers, preventing microphone feedback during live
    translation sessions.

    The VitsModel and VitsTokenizer are loaded inside run() (on the TTSThread)
    to keep __init__ cheap so the Pipeline constructor stays fast.
    """

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
        self._device: str = config.MMS_TTS_DEVICE

        # Model and tokenizer loaded in run() — do NOT load here
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None

    # ── Thread target ──────────────────────────────────────────────────────

    def run(self) -> None:
        """Load MMS-TTS, then synthesise translated segments into WAV files."""
        # ── Resolve "auto" device ──────────────────────────────────────────
        import torch  # noqa: PLC0415
        if self._device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(
            "Loading MMS-TTS model '%s' on device=%s…",
            config.MMS_TTS_MODEL_ID,
            self._device,
        )

        from transformers import VitsModel, VitsTokenizer  # noqa: PLC0415

        self._tokenizer = VitsTokenizer.from_pretrained(config.MMS_TTS_MODEL_ID)
        self._model = VitsModel.from_pretrained(config.MMS_TTS_MODEL_ID)
        self._model = self._model.to(torch.device(self._device))
        self._model.eval() # type: ignore

        self._output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "TTS engine ready (MMS-TTS, device=%s, model=%s)",
            self._device,
            config.MMS_TTS_MODEL_ID,
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
            success = self._synthesise(item.text, out_path)
            elapsed = time.perf_counter() - t0

            # End-to-end latency measured from audio capture to file write
            e2e = time.perf_counter() - item.capture_timestamp
            if success:
                print(f"[TTS   #{item.chunk_id:>4d}] saved → {out_path}", flush=True)
                logger.debug(
                    "TTS saved (%.3fs synthesis, %.3fs e2e) chunk #%d → %s",
                    elapsed,
                    e2e,
                    item.chunk_id,
                    out_path,
                )
            else:
                print(f"[TTS   #{item.chunk_id:>4d}] synthesis failed — no file written", flush=True)

        logger.info("TTSEngine stopped.")

    # ── Helpers ────────────────────────────────────────────────────────────

    def _synthesise(
        self,
        text: str,
        path: Path,
    ) -> bool:
        """Synthesise ``text`` via MMS-TTS and write a WAV file to ``path``.

        Args:
            text: Hindi (or target-language) text to synthesise.
            path: Destination WAV file path.
        """
        if self._model is None or self._tokenizer is None:
            logger.warning("TTS model not loaded — skipping synthesis for %s.", path.name)
            return False

        try:
            import torch  # noqa: PLC0415

            inputs = self._tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                output = self._model(**inputs)

            waveform = output.waveform[0].squeeze().cpu().numpy()
            sf.write(
                str(path),
                waveform.astype(np.float32),
                samplerate=self._model.config.sampling_rate,
            )
            return True

        except Exception:
            logger.exception("MMS-TTS synthesis failed for chunk '%s'.", path.name)
            return False
