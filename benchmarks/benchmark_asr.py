"""
benchmark_asr.py — ASR model comparison benchmark.

Compares faster-whisper model sizes (tiny.en, base.en, small.en) on
synthesized test clips, measuring Word Error Rate (WER) and latency.

Usage
-----
    python benchmarks/benchmark_asr.py

Output
------
    results/asr_results.csv
"""

from __future__ import annotations

import csv
import os
import sys
import time

# Project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config  # noqa: E402  — must be first project import

import numpy as np  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
GROUND_TRUTH = os.path.join(PROJECT_ROOT, "test_clips", "ground_truth.txt")
CLIPS_DIR = os.path.join(PROJECT_ROOT, "test_clips")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "asr_results.csv")

MODEL_SIZES = ["tiny.en", "base.en", "small.en"]


# ── WER calculation ─────────────────────────────────────────────────────────

def _word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute WER using Levenshtein distance on word sequences."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    # Dynamic programming — standard edit distance
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


# ── Helpers ──────────────────────────────────────────────────────────────────

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
    """Load WAV file as float32 numpy array at 16 kHz."""
    import scipy.io.wavfile as wavfile

    sr, data = wavfile.read(wav_path)
    # Convert to float32 in [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)

    # Convert stereo to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample to 16 kHz if needed
    if sr != 16000:
        from scipy.signal import resample
        num_samples = int(len(data) * 16000 / sr)
        data = resample(data, num_samples).astype(np.float32)

    return data


# ── Main benchmark ───────────────────────────────────────────────────────────

def run_benchmark() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    entries = _load_ground_truth()
    print(f"Loaded {len(entries)} test clips from ground_truth.txt\n")

    # Pre-load all audio
    audio_data: dict[str, np.ndarray] = {}
    for fname, _text in entries:
        wav_path = os.path.join(CLIPS_DIR, fname)
        if not os.path.exists(wav_path):
            print(f"  ⚠ Missing: {wav_path} — run generate_test_clips.py first")
            continue
        audio_data[fname] = _load_audio(wav_path)
    print(f"Pre-loaded {len(audio_data)} audio files.\n")

    if not audio_data:
        print("ERROR: No audio files found. Run generate_test_clips.py first.")
        sys.exit(1)

    all_rows: list[dict] = []
    summary_rows: list[dict] = []

    for model_size in MODEL_SIZES:
        print(f"{'=' * 60}")
        print(f"  Model: {model_size}")
        print(f"{'=' * 60}")

        # Load model
        t0 = time.perf_counter()
        model = WhisperModel(
            model_size,
            device=config.ASR_DEVICE,
            compute_type=config.ASR_COMPUTE_TYPE,
        )
        load_time = time.perf_counter() - t0
        print(f"  Model loaded in {load_time:.1f}s\n")

        wers = []
        latencies = []

        for fname, ground_truth in entries:
            if fname not in audio_data:
                continue

            audio = audio_data[fname]

            # Transcribe
            t0 = time.perf_counter()
            segments_gen, _info = model.transcribe(
                audio,
                beam_size=config.ASR_BEAM_SIZE,
                language=config.ASR_LANGUAGE,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            hypothesis = "".join(seg.text for seg in segments_gen).strip()
            latency = time.perf_counter() - t0

            wer = _word_error_rate(ground_truth, hypothesis)
            wers.append(wer)
            latencies.append(latency)

            all_rows.append({
                "model": model_size,
                "clip": fname,
                "ground_truth": ground_truth,
                "hypothesis": hypothesis,
                "wer": f"{wer:.4f}",
                "latency_s": f"{latency:.4f}",
            })

            print(f"  {fname}: WER={wer:.2%}  latency={latency:.3f}s")
            print(f"    REF: {ground_truth}")
            print(f"    HYP: {hypothesis}")

        # Model summary
        avg_wer = np.mean(wers) if wers else 0
        avg_lat = np.mean(latencies) if latencies else 0
        std_lat = np.std(latencies) if latencies else 0

        summary_rows.append({
            "model": model_size,
            "avg_wer": f"{avg_wer:.4f}",
            "avg_latency_s": f"{avg_lat:.4f}",
            "std_latency_s": f"{std_lat:.4f}",
            "num_clips": len(wers),
        })

        print(f"\n  ── Summary for {model_size} ──")
        print(f"  Average WER:     {avg_wer:.2%}")
        print(f"  Average latency: {avg_lat:.3f}s ± {std_lat:.3f}s")
        print()

        # Release model memory
        del model

    # ── Write CSV ────────────────────────────────────────────────────────────
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "clip", "ground_truth", "hypothesis", "wer", "latency_s",
        ])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n✓ Detailed results saved to {RESULTS_CSV}")

    # Summary CSV
    summary_csv = os.path.join(RESULTS_DIR, "asr_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "avg_wer", "avg_latency_s", "std_latency_s", "num_clips",
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"✓ Summary saved to {summary_csv}")

    # Print paper-ready table
    print(f"\n{'=' * 60}")
    print("  ASR Results Table (for Paper Section 5)")
    print(f"{'=' * 60}")
    print(f"  {'Model':<12} {'Avg WER':>10} {'Avg Latency':>14} {'Std Latency':>14}")
    print(f"  {'─' * 12} {'─' * 10} {'─' * 14} {'─' * 14}")
    for row in summary_rows:
        print(f"  {row['model']:<12} {float(row['avg_wer']):>9.2%} {row['avg_latency_s']:>13}s {row['std_latency_s']:>13}s")


if __name__ == "__main__":
    run_benchmark()
