"""
tests/test_tts.py — Verify MMS-TTS engine initialisation and synthesis.

Tests synthesise a short Hindi phrase and check that a valid WAV file
(at model.config.sampling_rate) is produced.  No reference audio is
necessary — MMS-TTS (facebook/mms-tts-hin) is a fixed-voice model.

Run:
    python -m pytest tests/test_tts.py -v
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config  # noqa: F401

import pytest

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

_SAMPLE_TEXT = "नमस्ते, यह एक परीक्षण है।"  # "Hello, this is a test."

# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _patch_nfe_step():
    """Reduce diffusion steps 32 → 8 for all TTS tests (~4× faster synthesis)."""
    import f5_tts.infer.utils_infer as _utils  # noqa: PLC0415

    original = _utils.nfe_step
    _utils.nfe_step = 8
    yield
    _utils.nfe_step = original


@pytest.fixture(scope="session")
def indicf5_model():
    """Load IndicF5 once for the entire test session (avoids repeated 1.6 GB disk reads)."""
    if not _ref_audio_available():
        pytest.skip(
            f"IndicF5 reference audio not found at '{_REF_AUDIO}'. "
            "Run 'python setup_models.py' first."
        )
    from transformers import AutoModel  # noqa: PLC0415

    return AutoModel.from_pretrained(config.INDICF5_MODEL_ID, trust_remote_code=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mms_tts_model_loads() -> None:
    """VitsModel.from_pretrained should load without error."""
    from transformers import VitsModel  # noqa: PLC0415

    model = VitsModel.from_pretrained(config.MMS_TTS_MODEL_ID)
    assert model is not None, "VitsModel.from_pretrained returned None"
    print(f"✓ test_mms_tts_model_loads passed (model={config.MMS_TTS_MODEL_ID})")


def test_mms_tts_synthesises_wav(tmp_path: Path) -> None:
    """MMS-TTS should produce a non-empty float32 WAV at model.config.sampling_rate."""
    import torch  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415
    from transformers import VitsModel, VitsTokenizer  # noqa: PLC0415

    tokenizer = VitsTokenizer.from_pretrained(config.MMS_TTS_MODEL_ID)
    model = VitsModel.from_pretrained(config.MMS_TTS_MODEL_ID)
    model.eval()

    inputs = tokenizer(_SAMPLE_TEXT, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)

    waveform = output.waveform[0].squeeze().cpu().numpy()
    sr = model.config.sampling_rate

    out_path = tmp_path / "test_synthesis.wav"
    sf.write(str(out_path), waveform.astype(np.float32), samplerate=sr)

    assert out_path.exists(), "Output WAV file was not created"
    data, file_sr = sf.read(str(out_path))
    assert file_sr == sr, f"Expected {sr} Hz, got {file_sr}"
    assert len(data) > 0, "Output WAV is empty"
    print(f"✓ test_mms_tts_synthesises_wav passed  (duration={len(data)/sr:.2f}s, sr={sr} Hz)")


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

    # Allow time for MMS-TTS model to load
    time.sleep(5.0)

    seg = TranslatedSegment(chunk_id=42, text=_SAMPLE_TEXT)
    tts_q.put(seg)
    tts_q.put(None)  # sentinel

    # MMS-TTS (VITS, non-autoregressive) is much faster than IndicF5
    t.join(timeout=60)
    assert not t.is_alive(), "TTSEngine thread did not exit within 60 s"

    if "error" in results:
        pytest.fail(f"TTSEngine.run() raised: {results['error']}")
    assert results.get("ok"), "TTSEngine did not signal success"

    out_wav = tmp_path / "chunk_0042.wav"
    assert out_wav.exists(), f"Expected output WAV not found: {out_wav}"
    print(f"✓ test_tts_engine_class passed  (output={out_wav})")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
