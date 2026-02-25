"""
setup_models.py — One-time model download and verification script.

Run this script ONCE while online to download all required models.
After that, PolyglotTalk runs fully offline.

What it downloads
-----------------
1. faster-whisper base.en (int8) — ~150 MB
   Cached to:  ~/.cache/huggingface/hub/  (or WHISPER_MODELS_DIR)

2. Argos Translate en→hi language pack — ~100 MB
   Installed to:  ~/.local/share/argos-translate/packages/  (Linux)
                  %LOCALAPPDATA%\\argos-translate\\packages\\  (Windows)

3. AI4Bharat IndicF5 TTS model — ~400 MB (float32)
   Cached to:  ~/.cache/huggingface/hub/models--ai4bharat--IndicF5/
   Hindi reference audio prompt saved to:  prompts/HIN_F_HAPPY_00001.wav

Total:  ~650-750 MB depending on cached HuggingFace files.

Usage
-----
    python setup_models.py
    python setup_models.py --skip-verify   # download only, skip smoke tests
"""

from __future__ import annotations

# config MUST be imported first — sets OMP_NUM_THREADS / CT2_INTER_THREADS
import config  # noqa: F401

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
    print("\n[2/3] Downloading Argos Translate language pack…")
    _info(f"Language pair: {config.SOURCE_LANG} → {config.TARGET_LANG}")

    import argostranslate.package

    # Check whether the package is already installed
    installed = argostranslate.package.get_installed_packages()
    already = any(
        p.from_code == config.SOURCE_LANG and p.to_code == config.TARGET_LANG
        for p in installed
    )
    if already:
        _ok(f"Argos package {config.SOURCE_LANG}→{config.TARGET_LANG} already installed.")
        return

    _info("Fetching package index (requires internet)…")
    argostranslate.package.update_package_index()

    available = argostranslate.package.get_available_packages()
    pkg = next(
        (
            p
            for p in available
            if p.from_code == config.SOURCE_LANG and p.to_code == config.TARGET_LANG
        ),
        None,
    )
    if pkg is None:
        _fail(
            f"No Argos package found for {config.SOURCE_LANG}→{config.TARGET_LANG}. "
            f"Check https://www.argosopentech.com/argospm/index/ for available pairs."
        )
        sys.exit(1)

    _info(f"Downloading {pkg.from_name} → {pkg.to_name} (version {pkg.package_version})…")
    t0 = time.perf_counter()
    download_path = pkg.download()
    argostranslate.package.install_from_path(download_path)
    elapsed = time.perf_counter() - t0
    _ok(f"Argos package installed in {elapsed:.1f}s  →  {download_path}")


def verify_translation_model() -> None:
    _info('Smoke-testing translation model ("Hello")…')
    import argostranslate.translate

    result = argostranslate.translate.translate(
        "Hello", config.SOURCE_LANG, config.TARGET_LANG
    )
    if not result or not result.strip():
        _fail("Translation smoke test failed — empty output!")
        sys.exit(1)
    _ok(f"Translation smoke test passed: \"Hello\" → \"{result.strip()}\"")


# ── Step 3: AI4Bharat IndicF5 TTS ───────────────────────────────────────────

def download_tts_model() -> None:
    """Download IndicF5 model weights and a Hindi reference audio prompt."""
    print("\n[3/3] Downloading AI4Bharat IndicF5 TTS model…")
    _info(f"Model: {config.INDICF5_MODEL_ID}  device: {config.INDICF5_DEVICE}")

    from huggingface_hub import snapshot_download  # noqa: PLC0415

    # Pre-download all model files to cache WITHOUT instantiating the model.
    # Instantiation (which loads the vocoder) happens later in TTSThread
    # where it doesn't conflict with transformers' meta-tensor initialization.
    t0 = time.perf_counter()
    _cache_dir = snapshot_download(config.INDICF5_MODEL_ID)
    elapsed = time.perf_counter() - t0
    _ok(f"IndicF5 model files downloaded to cache in {elapsed:.1f}s")

    # ── Download Hindi reference audio prompt ─────────────────────────────
    from pathlib import Path as _Path  # noqa: PLC0415
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    ref_dest = _Path(config.INDICF5_REF_AUDIO_PATH)
    if ref_dest.exists():
        _ok(f"Hindi reference audio already present: {ref_dest}")
        return

    ref_dest.parent.mkdir(parents=True, exist_ok=True)
    ref_filename = ref_dest.name  # e.g. HIN_F_HAPPY_00001.wav

    _info(f"Downloading reference audio '{ref_filename}' from {config.INDICF5_MODEL_ID}…")
    
    try:
        downloaded = hf_hub_download(
            repo_id=config.INDICF5_MODEL_ID,
            filename=f"prompts/{ref_filename}",
        )
        import shutil  # noqa: PLC0415
        shutil.copy2(downloaded, ref_dest)
        _ok(f"Hindi reference audio saved → {ref_dest}")

    except Exception as exc:  # pylint: disable=broad-except
        # Fallback: try the Punjabi prompt that ships in the README example
        _info(f"'{ref_filename}' not found ({exc}); attempting Punjabi fallback…")
        try:
            fallback_file = "PAN_F_HAPPY_00001.wav"
            downloaded = hf_hub_download(
                repo_id=config.INDICF5_MODEL_ID,
                filename=f"prompts/{fallback_file}",
            )
            import shutil  # noqa: PLC0415
            shutil.copy2(downloaded, ref_dest)
            _ok(f"Fallback Punjabi reference audio saved → {ref_dest}")
        except Exception as exc2:  # pylint: disable=broad-except
            _fail(
                f"Could not download any reference audio: {exc2}\n"
                "   Please manually place a short (~10s) Hindi WAV file at "
                f"{ref_dest} and re-run."
            )


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
