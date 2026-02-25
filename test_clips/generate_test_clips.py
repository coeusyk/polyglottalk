"""
generate_test_clips.py — Synthesize WAV test clips from ground_truth.txt using pyttsx3.

Produces 25 mono 16 kHz WAV clips (~2.5 s each) with known transcriptions
so WER can be calculated against exact ground truth.

Usage
-----
    python test_clips/generate_test_clips.py
"""

from __future__ import annotations

import os
import sys
import wave
import struct
import threading

# Add project root to path so config can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config  # noqa: E402

import pyttsx3  # noqa: E402


GROUND_TRUTH_FILE = os.path.join(os.path.dirname(__file__), "ground_truth.txt")
OUTPUT_DIR = os.path.dirname(__file__)


def _parse_ground_truth() -> list[tuple[str, str]]:
    """Parse ground_truth.txt → list of (filename, transcription)."""
    entries = []
    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            filename, text = line.split("|", 1)
            entries.append((filename.strip(), text.strip()))
    return entries


def _generate_clips(entries: list[tuple[str, str]]) -> None:
    """Generate WAV files using pyttsx3 in a dedicated thread (COM requirement)."""
    errors: list[str] = []

    def _worker():
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)  # Slower rate for clarity

            for filename, text in entries:
                outpath = os.path.join(OUTPUT_DIR, filename)
                # pyttsx3 save_to_file produces a WAV
                engine.save_to_file(text, outpath)
                engine.runAndWait()
                if os.path.exists(outpath):
                    size = os.path.getsize(outpath)
                    print(f"  ✓ {filename} ({size:,} bytes) — \"{text}\"")
                else:
                    errors.append(filename)
                    print(f"  ✗ {filename} — file not created!")

            engine.stop()
        except Exception as exc:
            errors.append(f"Engine error: {exc}")

    t = threading.Thread(target=_worker)
    t.start()
    t.join(timeout=120)

    if errors:
        print(f"\n⚠ {len(errors)} clip(s) failed to generate.")
    else:
        print(f"\n✓ All {len(entries)} clips generated successfully in {OUTPUT_DIR}")


def main() -> None:
    print("=" * 60)
    print(" PolyglotTalk — Test Clip Generator")
    print("=" * 60)

    entries = _parse_ground_truth()
    print(f"\nGenerating {len(entries)} clips from ground_truth.txt...\n")
    _generate_clips(entries)


if __name__ == "__main__":
    main()
