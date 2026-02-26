"""
benchmark_e2e.py — End-to-end pipeline latency benchmark.

Measures per-stage latency (ASR, MT, TTS) and total E2E time by
feeding test clips through each stage sequentially.

Usage
-----
    python benchmarks/benchmark_e2e.py

Output
------
    results/e2e_latency.csv
"""

from __future__ import annotations

import csv
import os
import sys
import threading
import time

# Project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config  # noqa: E402

import numpy as np  # noqa: E402

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
GROUND_TRUTH = os.path.join(PROJECT_ROOT, "test_clips", "ground_truth.txt")
CLIPS_DIR = os.path.join(PROJECT_ROOT, "test_clips")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "e2e_latency.csv")

NUM_TRIALS = 20


def _load_ground_truth() -> list[tuple[str, str]]:
    """Return list of (wav_filename, transcription) from ground_truth.txt."""
    entries = []
    with open(GROUND_TRUTH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fname, text = line.split("|", 1)
            entries.append((fname.strip(), text.strip()))
    return entries


def _load_audio(wav_path: str) -> np.ndarray:
    """Load WAV as float32 16 kHz mono."""
    import scipy.io.wavfile as wavfile

    sr, data = wavfile.read(wav_path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)

    if data.ndim > 1:
        data = data.mean(axis=1)

    if sr != 16000:
        from scipy.signal import resample
        num_samples = int(len(data) * 16000 / sr)
        data = resample(data, num_samples).astype(np.float32)

    return data


def run_benchmark() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    entries = _load_ground_truth()
    print(f"Loaded {len(entries)} test clips.\n")

    # Pre-load audio
    audio_clips: list[tuple[str, np.ndarray]] = []
    for fname, _text in entries:
        wav_path = os.path.join(CLIPS_DIR, fname)
        if os.path.exists(wav_path):
            audio_clips.append((fname, _load_audio(wav_path)))

    if not audio_clips:
        print("ERROR: No audio files found. Run generate_test_clips.py first.")
        sys.exit(1)

    print(f"Pre-loaded {len(audio_clips)} audio files.\n")

    # ── Load models (once) ───────────────────────────────────────────────────
    print("Loading ASR model...")
    from faster_whisper import WhisperModel
    asr_model = WhisperModel(
        config.ASR_MODEL_SIZE,
        device=config.ASR_DEVICE,
        compute_type=config.ASR_COMPUTE_TYPE,
    )
    print(f"  ✓ ASR model loaded ({config.ASR_MODEL_SIZE})")

    print("Loading translation model...")
    import argostranslate.translate
    # Verify Argos model is installed
    import argostranslate.package
    installed = argostranslate.package.get_installed_packages()
    if not any(p.from_code == "en" and p.to_code == "hi" for p in installed):
        print("ERROR: Argos en→hi package not installed. Run setup_models.py.")
        sys.exit(1)
    print(f"  ✓ Translation model loaded (Argos en→hi)")

    print("Preparing TTS engine...")
    import pyttsx3
    # TTS must be used in the thread that creates it (COM/SAPI5 on Windows)

    print(f"\nRunning {NUM_TRIALS} E2E trials...\n")

    # ── Run trials ───────────────────────────────────────────────────────────
    all_rows: list[dict] = []

    for trial in range(1, NUM_TRIALS + 1):
        # Pick clip (cycle through available clips)
        clip_idx = (trial - 1) % len(audio_clips)
        fname, audio = audio_clips[clip_idx]

        # ── ASR stage ────────────────────────────────────────────────────────
        t_asr_start = time.perf_counter()
        segments_gen, _info = asr_model.transcribe(
            audio,
            beam_size=config.ASR_BEAM_SIZE,
            language=config.ASR_LANGUAGE,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        transcript = "".join(seg.text for seg in segments_gen).strip()
        t_asr_end = time.perf_counter()
        asr_time = t_asr_end - t_asr_start

        if not transcript:
            transcript = "hello"  # fallback for empty transcripts

        # ── MT stage ─────────────────────────────────────────────────────────
        t_mt_start = time.perf_counter()
        translated = argostranslate.translate.translate(transcript, "en", "hi")
        t_mt_end = time.perf_counter()
        mt_time = t_mt_end - t_mt_start

        # ── TTS stage ────────────────────────────────────────────────────────
        # Run TTS in a separate thread (COM requirement on Windows)
        tts_time_result = [0.0]
        tts_error = [None]

        def _tts_worker():
            try:
                engine = pyttsx3.init()
                # Use an explicit TTS speaking rate for pyttsx3 (approx. default).
                engine.setProperty("rate", 180)
                t_tts_start = time.perf_counter()
                engine.say(translated)
                engine.runAndWait()
                tts_time_result[0] = time.perf_counter() - t_tts_start
                engine.stop()
            except Exception as exc:
                tts_error[0] = exc
                tts_time_result[0] = 0.0

        tts_thread = threading.Thread(target=_tts_worker)
        tts_thread.start()
        tts_thread.join(timeout=30)

        tts_time = tts_time_result[0]
        total_e2e = asr_time + mt_time + tts_time

        all_rows.append({
            "trial": trial,
            "clip": fname,
            "transcript": transcript,
            "translation": translated,
            "asr_time_s": f"{asr_time:.4f}",
            "mt_time_s": f"{mt_time:.4f}",
            "tts_time_s": f"{tts_time:.4f}",
            "total_e2e_s": f"{total_e2e:.4f}",
        })

        print(f"  Trial {trial:>2}: ASR={asr_time:.3f}s  MT={mt_time:.3f}s  "
              f"TTS={tts_time:.3f}s  E2E={total_e2e:.3f}s")

    # ── Statistics ───────────────────────────────────────────────────────────
    asr_times = [float(r["asr_time_s"]) for r in all_rows]
    mt_times = [float(r["mt_time_s"]) for r in all_rows]
    tts_times = [float(r["tts_time_s"]) for r in all_rows]
    e2e_times = [float(r["total_e2e_s"]) for r in all_rows]

    stats = {
        "asr": (np.mean(asr_times), np.std(asr_times)),
        "mt": (np.mean(mt_times), np.std(mt_times)),
        "tts": (np.mean(tts_times), np.std(tts_times)),
        "e2e": (np.mean(e2e_times), np.std(e2e_times)),
    }

    # ── Write CSV ────────────────────────────────────────────────────────────
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "trial", "clip", "transcript", "translation",
            "asr_time_s", "mt_time_s", "tts_time_s", "total_e2e_s",
        ])
        writer.writeheader()
        writer.writerows(all_rows)

        # Summary row
        writer.writerow({
            "trial": "MEAN",
            "clip": "",
            "transcript": "",
            "translation": "",
            "asr_time_s": f"{stats['asr'][0]:.4f}",
            "mt_time_s": f"{stats['mt'][0]:.4f}",
            "tts_time_s": f"{stats['tts'][0]:.4f}",
            "total_e2e_s": f"{stats['e2e'][0]:.4f}",
        })
        writer.writerow({
            "trial": "STD",
            "clip": "",
            "transcript": "",
            "translation": "",
            "asr_time_s": f"{stats['asr'][1]:.4f}",
            "mt_time_s": f"{stats['mt'][1]:.4f}",
            "tts_time_s": f"{stats['tts'][1]:.4f}",
            "total_e2e_s": f"{stats['e2e'][1]:.4f}",
        })

    print(f"\n✓ Results saved to {RESULTS_CSV}")

    # Print paper-ready table
    print(f"\n{'=' * 60}")
    print("  E2E Latency Table (for Paper Section 5)")
    print(f"{'=' * 60}")
    print(f"  {'Stage':<12} {'Mean (s)':>12} {'Std (s)':>12}")
    print(f"  {'─' * 12} {'─' * 12} {'─' * 12}")
    for stage, (mean, std) in stats.items():
        print(f"  {stage.upper():<12} {mean:>12.4f} {std:>12.4f}")


if __name__ == "__main__":
    run_benchmark()
