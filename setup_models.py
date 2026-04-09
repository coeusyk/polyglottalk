"""
setup_models.py — One-time model download and verification script.

Run this script ONCE while online to download all required models.
After that, PolyglotTalk runs fully offline.

What it downloads
-----------------
1. faster-whisper base.en (int8) — ~150 MB
   Cached to:  ~/.cache/huggingface/hub/  (or WHISPER_MODELS_DIR)

2. Argos Translate language packs for all languages in ARGOS_LANG_MAP — ~100 MB each
   Installed to:  ~/.local/share/argos-translate/packages/  (Linux)
                  %LOCALAPPDATA%\\argos-translate\\packages\\  (Windows)

3. Facebook MMS-TTS model for TARGET_LANG — ~150 MB
   Cached to:  ~/.cache/huggingface/hub/models--facebook--mms-tts-{lang}/

Total:  ~400-650 MB depending on cached HuggingFace files and language selection.

Usage
-----
    python setup_models.py
    python setup_models.py --skip-verify   # download only, skip smoke tests
"""

from __future__ import annotations

# config MUST be imported first — sets OMP_NUM_THREADS / CT2_INTER_THREADS
from polyglot_talk import config  # noqa: F401

import argparse
import sys
import time


# ── ANSI helpers ─────────────────────────────────────────────────────────────

def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _info(msg: str) -> None:
    print(f"  → {msg}")


def _fail(msg: str) -> None:
    print(f"  ✗ {msg}", file=sys.stderr)


# ── Step 1: faster-whisper ────────────────────────────────────────────────────

def download_asr_model():
    print("\n[1/3] Downloading faster-whisper model…")
    _info(f"Model: {config.ASR_MODEL_SIZE}  compute: {config.ASR_COMPUTE_TYPE}  device: {config.ASR_DEVICE}")

    from faster_whisper import WhisperModel

    t0 = time.perf_counter()
    model = WhisperModel(
        config.ASR_MODEL_SIZE,
        device=config.ASR_DEVICE,
        compute_type=config.ASR_COMPUTE_TYPE,
    )
    
    elapsed = time.perf_counter() - t0
    _ok(f"faster-whisper model loaded/verified in {elapsed:.1f}s")

    return model


def verify_asr_model(model) -> None:
    _info("Smoke-testing ASR model (1 second of silence)…")
    import numpy as np

    silence = np.zeros(config.SAMPLE_RATE, dtype="float32")  # 1 s silence
    segments_gen, _info_obj = model.transcribe(
        silence,
        beam_size=config.ASR_BEAM_SIZE,
        language=config.ASR_LANGUAGE,
        vad_filter=False,
    )
    # Drain generator fully — required by faster-whisper
    _ = list(segments_gen)
    _ok("ASR smoke test passed (silence → no crash)")


# ── Step 2: Argos Translate ──────────────────────────────────────────────────

def download_translation_model() -> None:
    print("\n[2/3] Downloading Argos Translate language packs…")

    import argostranslate.package

    _info("Fetching package index (requires internet)…")
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()

    for argos_code in config.ARGOS_LANG_MAP.values():
        pair_label = f"{config.SOURCE_LANG}→{argos_code}"
        installed = argostranslate.package.get_installed_packages()
        already = any(
            p.from_code == config.SOURCE_LANG and p.to_code == argos_code
            for p in installed
        )
        if already:
            _ok(f"Argos package {pair_label} already installed — skipping.")
            continue

        pkg = next(
            (
                p
                for p in available
                if p.from_code == config.SOURCE_LANG and p.to_code == argos_code
            ),
            None,
        )
        if pkg is None:
            _fail(
                f"No Argos package found for {pair_label}. "
                f"Check https://www.argosopentech.com/argospm/index/ for available pairs."
            )
            sys.exit(1)

        _info(f"Downloading {pkg.from_name} → {pkg.to_name} (version {pkg.package_version})…")
        t0 = time.perf_counter()
        download_path = pkg.download()
        argostranslate.package.install_from_path(download_path)
        elapsed = time.perf_counter() - t0
        _ok(f"Argos package {pair_label} installed in {elapsed:.1f}s  →  {download_path}")


def verify_translation_model() -> None:
    _info('Smoke-testing translation model ("Hello")…')
    import argostranslate.translate

    argos_target = config.ARGOS_LANG_MAP[config.TARGET_LANG]
    result = argostranslate.translate.translate(
        "Hello", config.SOURCE_LANG, argos_target
    )
    if not result or not result.strip():
        _fail("Translation smoke test failed — empty output!")
        sys.exit(1)
    _ok(f"Translation smoke test passed: \"Hello\" → \"{result.strip()}\"")


# ── Step 3: Facebook MMS-TTS ─────────────────────────────────────────────────

def download_tts_model() -> None:
    """Download MMS-TTS model weights to the HuggingFace cache."""
    print("\n[3/3] Downloading Facebook MMS-TTS model…")
    model_id = config.MMS_TTS_MODEL_MAP[config.TARGET_LANG]
    _info(f"Model: {model_id}  device: {config.MMS_TTS_DEVICE}")

    from transformers import VitsModel, VitsTokenizer  # noqa: PLC0415

    t0 = time.perf_counter()
    # Pre-download + verify by loading tokenizer and model weights.
    # The model is not moved to the target device here — that happens
    # lazily in TTSEngine.run() on the dedicated TTS thread.
    _tokenizer = VitsTokenizer.from_pretrained(model_id)
    _model = VitsModel.from_pretrained(model_id)
    elapsed = time.perf_counter() - t0
    del _tokenizer, _model  # free memory — just needed for cache warm-up
    _ok(f"MMS-TTS model downloaded/verified in {elapsed:.1f}s")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PolyglotTalk model setup")
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip smoke tests (download-only mode)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" PolyglotTalk — Model Setup")
    print("=" * 60)

    asr_model = download_asr_model()
    if not args.skip_verify:
        verify_asr_model(asr_model)

    download_translation_model()
    if not args.skip_verify:
        verify_translation_model()

    download_tts_model()

    print("\n" + "=" * 60)
    print(" ✓ All models ready for offline use.")
    print(" Run 'python main.py' to start PolyglotTalk.")
    print("=" * 60)


if __name__ == "__main__":
    main()
