"""
setup_models.py — One-time model download and verification script.

Run this script ONCE while online to download all required models.
After that, PolyglotTalk runs fully offline.

What it downloads
-----------------
1. faster-whisper base.en (int8) — ~150 MB
   Cached to:  ~/.cache/huggingface/hub/  (or WHISPER_MODELS_DIR)

2a. Argos Translate en→hi — ~100 MB  [Hindi only]
    Installed to:  ~/.local/share/argos-translate/packages/  (Linux)
                   %LOCALAPPDATA%\\argos-translate\\packages\\  (Windows)

2b. Helsinki-NLP MarianMT checkpoint for the active TARGET_LANG — ~300 MB
    [all non-Hindi languages]
    Cached to:  ~/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-en-{xx}/

3.  Facebook MMS-TTS model for TARGET_LANG — ~150 MB
    Cached to:  ~/.cache/huggingface/hub/models--facebook--mms-tts-{lang}/

Total:  ~400-650 MB depending on cached HuggingFace files and language selection.

Note on MT backend
------------------
Argos Translate only publishes an en→hi offline package for Indian languages.
All other Indian language pairs use Helsinki-NLP MarianMT via transformers.
This script installs whichever backend applies to config.TARGET_LANG.

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
    print(f"  \u2713 {msg}")


def _info(msg: str) -> None:
    print(f"  \u2192 {msg}")


def _warn(msg: str) -> None:
    print(f"  ! {msg}", file=sys.stderr)


def _fail(msg: str) -> None:
    print(f"  \u2717 {msg}", file=sys.stderr)


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

    silence = np.zeros(config.SAMPLE_RATE, dtype="float32")
    segments_gen, _info_obj = model.transcribe(
        silence,
        beam_size=config.ASR_BEAM_SIZE,
        language=config.ASR_LANGUAGE,
        vad_filter=False,
    )
    _ = list(segments_gen)
    _ok("ASR smoke test passed (silence → no crash)")


# ── Step 2a: Argos Translate (Hindi only) ────────────────────────────────────

def download_argos_model() -> bool:
    """Attempt to install the Argos en→hi package.

    Returns True if the package was already installed or successfully
    downloaded.  Returns False (with a warning) if the package is not
    found in the upstream index — this is non-fatal because Hindi may
    still work if a prior install exists, and all other languages use
    MarianMT.
    """
    argos_code = config.ARGOS_LANG_MAP.get(config.TARGET_LANG)
    if argos_code is None:
        # TARGET_LANG is not in ARGOS_LANG_MAP — Argos is not used at all.
        return True

    print("\n[2a/3] Downloading Argos Translate language pack (Hindi)…")

    import argostranslate.package

    _info("Fetching Argos package index (requires internet)…")
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()

    pair_label = f"{config.SOURCE_LANG}→{argos_code}"

    installed = argostranslate.package.get_installed_packages()
    already = any(
        p.from_code == config.SOURCE_LANG and p.to_code == argos_code
        for p in installed
    )
    if already:
        _ok(f"Argos package {pair_label} already installed — skipping.")
        return True

    pkg = next(
        (
            p
            for p in available
            if p.from_code == config.SOURCE_LANG and p.to_code == argos_code
        ),
        None,
    )
    if pkg is None:
        # Non-fatal: warn and continue.  The pipeline will raise at startup
        # if it actually tries to use Argos without the package.
        _warn(
            f"No Argos package found for {pair_label} in the upstream index.\n"
            f"  Check https://www.argosopentech.com/argospm/index/ for available pairs.\n"
            f"  If you need Hindi TTS, install the package manually and re-run this script."
        )
        return False

    _info(f"Downloading {pkg.from_name} → {pkg.to_name} (version {pkg.package_version})…")
    t0 = time.perf_counter()
    download_path = pkg.download()
    argostranslate.package.install_from_path(download_path)
    elapsed = time.perf_counter() - t0
    _ok(f"Argos package {pair_label} installed in {elapsed:.1f}s")
    return True


# ── Step 2b: MarianMT (all non-Hindi languages) ──────────────────────────────

def download_marian_model() -> None:
    """Download the Helsinki-NLP MarianMT checkpoint for TARGET_LANG.

    This is a no-op if TARGET_LANG is Hindi (handled by Argos).
    The checkpoint is downloaded to the HuggingFace hub cache and
    does not need to be re-downloaded on subsequent runs.
    """
    if config.TARGET_LANG not in config.MARIANMT_MODEL_MAP:
        # TARGET_LANG uses Argos, not MarianMT — nothing to do here.
        return

    model_id = config.MARIANMT_MODEL_MAP[config.TARGET_LANG]
    print("\n[2b/3] Downloading MarianMT translation model…")
    _info(f"Model: {model_id}")

    from transformers import MarianMTModel, MarianTokenizer

    t0 = time.perf_counter()
    _tokenizer = MarianTokenizer.from_pretrained(model_id)
    _model = MarianMTModel.from_pretrained(model_id)
    elapsed = time.perf_counter() - t0
    del _tokenizer, _model  # free memory — just needed for cache warm-up
    _ok(f"MarianMT model downloaded/verified in {elapsed:.1f}s")


def verify_translation_model() -> None:
    """Smoke-test whichever MT backend is active for TARGET_LANG."""
    _info('Smoke-testing translation model ("Hello")…')

    if config.MT_BACKEND == "argos":
        import argostranslate.translate
        argos_target = config.ARGOS_LANG_MAP[config.TARGET_LANG]
        result = argostranslate.translate.translate("Hello", config.SOURCE_LANG, argos_target)
    else:
        from transformers import pipeline as hf_pipeline
        import torch
        model_id = config.MARIANMT_MODEL_MAP[config.TARGET_LANG]
        device = 0 if torch.cuda.is_available() else -1
        pipe = hf_pipeline("translation", model=model_id, device=device)
        result = pipe("Hello")[0]["translation_text"]

    if not result or not result.strip():
        _fail("Translation smoke test failed — empty output!")
        sys.exit(1)
    _ok(f'Translation smoke test passed: "Hello" → "{result.strip()}"')


# ── Step 3: Facebook MMS-TTS ─────────────────────────────────────────────────

def download_tts_model() -> None:
    """Download MMS-TTS model weights to the HuggingFace cache."""
    print("\n[3/3] Downloading Facebook MMS-TTS model…")
    model_id = config.MMS_TTS_MODEL_MAP[config.TARGET_LANG]
    _info(f"Model: {model_id}  device: {config.MMS_TTS_DEVICE}")

    from transformers import VitsModel, VitsTokenizer  # noqa: PLC0415

    t0 = time.perf_counter()
    _tokenizer = VitsTokenizer.from_pretrained(model_id)
    _model = VitsModel.from_pretrained(model_id)
    elapsed = time.perf_counter() - t0
    del _tokenizer, _model
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
    print(f" Language: {config.TARGET_LANG}  MT backend: {config.MT_BACKEND}")
    print("=" * 60)

    asr_model = download_asr_model()
    if not args.skip_verify:
        verify_asr_model(asr_model)

    # MT backend: Argos (Hindi) or MarianMT (all other Indian languages)
    download_argos_model()    # no-op for non-Hindi; non-fatal if package missing
    download_marian_model()   # no-op for Hindi

    if not args.skip_verify:
        verify_translation_model()

    download_tts_model()

    print("\n" + "=" * 60)
    print(" \u2713 All models ready for offline use.")
    print(" Run 'python main.py' to start PolyglotTalk.")
    print("=" * 60)


if __name__ == "__main__":
    main()
