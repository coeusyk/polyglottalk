"""
tests/test_tts.py — Verify IndicF5 TTS engine initialisation and synthesis.

Tests synthesise a short Hindi phrase and check that a valid 24 kHz WAV
file is produced.  Each test is automatically skipped if the IndicF5
reference audio prompt has not yet been downloaded (run setup_models.py
first).

Run:
    python -m pytest tests/test_tts.py -v
"""

from __future__ import annotations

import os
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config  # noqa: F401

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REF_AUDIO = Path(config.INDICF5_REF_AUDIO_PATH)
_SAMPLE_TEXT = "नमस्ते, यह एक परीक्षण है।"  # "Hello, this is a test."


def _ref_audio_available() -> bool:
    return _REF_AUDIO.exists()


ref_audio_present = pytest.mark.skipif(
    not _ref_audio_available(),
    reason=(
        f"IndicF5 reference audio not found at '{_REF_AUDIO}'. "
        "Run 'python setup_models.py' first."
    ),
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@ref_audio_present
def test_indicf5_model_loads() -> None:
    """AutoModel.from_pretrained should load without error."""
    from transformers import AutoModel  # noqa: PLC0415

    model = AutoModel.from_pretrained(config.INDICF5_MODEL_ID, trust_remote_code=True)
    assert model is not None, "AutoModel.from_pretrained returned None"
    print("✓ test_indicf5_model_loads passed")


@ref_audio_present
def test_indicf5_synthesises_wav(tmp_path: Path) -> None:
    """IndicF5 should produce a non-empty float32 WAV at 24 kHz."""
    import numpy as np  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415
    from transformers import AutoModel  # noqa: PLC0415

    model = AutoModel.from_pretrained(config.INDICF5_MODEL_ID, trust_remote_code=True)

    audio = model(
        _SAMPLE_TEXT,
        ref_audio_path=str(_REF_AUDIO),
        ref_text=config.INDICF5_REF_TEXT,
    )

    if isinstance(audio, np.ndarray) and audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32_768.0

    out_path = tmp_path / "test_synthesis.wav"
    sf.write(str(out_path), np.array(audio, dtype=np.float32), samplerate=24_000)

    assert out_path.exists(), "Output WAV file was not created"
    data, sr = sf.read(str(out_path))
    assert sr == 24_000, f"Expected 24000 Hz, got {sr}"
    assert len(data) > 0, "Output WAV is empty"
    print(f"✓ test_indicf5_synthesises_wav passed  (duration={len(data)/sr:.2f}s)")


@ref_audio_present
def test_tts_engine_class(tmp_path: Path) -> None:
    """TTSEngine.run() must synthesise one segment and exit on sentinel."""
    import queue as _queue  # noqa: PLC0415
    from tts_engine import TTSEngine  # noqa: PLC0415
    from models import TranslatedSegment  # noqa: PLC0415

    results: dict[str, object] = {}

    tts_q: _queue.Queue = _queue.Queue()
    stop_event = threading.Event()

    engine = TTSEngine(
        tts_queue=tts_q,
        stop_event=stop_event,
        output_dir=str(tmp_path),
    )

    def _run() -> None:
        try:
            engine.run()
            results["ok"] = True
        except Exception as exc:  # pylint: disable=broad-except
            results["error"] = str(exc)

    t = threading.Thread(target=_run, name="TTSEngineTest", daemon=True)
    t.start()

    # Allow time for IndicF5 model to load ( ≤ 60 s on typical hardware)
    time.sleep(5.0)

    seg = TranslatedSegment(chunk_id=42, text=_SAMPLE_TEXT)
    tts_q.put(seg)
    tts_q.put(None)  # sentinel

    # IndicF5 first-call synthesis can take 10-60 s on CPU
    t.join(timeout=120)
    assert not t.is_alive(), "TTSEngine thread did not exit within 120 s"

    if "error" in results:
        pytest.fail(f"TTSEngine.run() raised: {results['error']}")
    assert results.get("ok"), "TTSEngine did not signal success"

    out_wav = tmp_path / "chunk_0042.wav"
    assert out_wav.exists(), f"Expected output WAV not found: {out_wav}"
    print(f"✓ test_tts_engine_class passed  (output={out_wav})")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
