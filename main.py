"""
main.py — Entry point for PolyglotTalk.

Usage
-----
    python main.py
    python main.py --source en --target hi
    python main.py --log-level DEBUG

IMPORTANT: config is imported first so that OMP_NUM_THREADS and
CT2_INTER_THREADS are set in os.environ before any CTranslate2 library
(faster-whisper, argostranslate) is imported anywhere in the process.
"""

from __future__ import annotations

# ── config MUST be the first project import ──────────────────────────────────
import config  # noqa: F401 — sets os.environ before CT2 is imported

import argparse
import logging
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="polyglot-talk",
        description="Real-time offline Speech-to-Speech Translation",
    )
    p.add_argument(
        "--source",
        default=config.SOURCE_LANG,
        metavar="LANG",
        help=f"Source language code (default: {config.SOURCE_LANG})",
    )
    p.add_argument(
        "--target",
        default=config.TARGET_LANG,
        metavar="LANG",
        help=f"Target language code (default: {config.TARGET_LANG})",
    )
    p.add_argument(
        "--log-level",
        default=config.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # ── Logging ───────────────────────────────────────────────────────────
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=config.LOG_FORMAT,
        stream=sys.stdout,
    )
    # Suppress faster-whisper's verbose "Processing audio with duration" messages
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)

    device = config.INDICF5_DEVICE
    if device == "auto":
        try:
            import torch  # noqa: PLC0415
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    print("=" * 60)
    print(" PolyglotTalk v0.1 — Offline Speech-to-Speech Translation")
    print(f" {args.source.upper()} → {args.target.upper()}  |  TTS: IndicF5 ({device})  |  No cloud APIs")
    print(f" TTS output saved to: {config.TTS_OUTPUT_DIR}/chunk_NNNN.wav")
    print("=" * 60)

    # ── Clean up old output chunks ──────────────────────────────────────
    output_dir = Path(config.TTS_OUTPUT_DIR)
    if output_dir.exists():
        for wav_file in output_dir.glob("chunk_*.wav"):
            wav_file.unlink()
            logging.getLogger(__name__).debug("Removed old chunk: %s", wav_file.name)

    # ── Import pipeline here so CT2 env vars are already set ─────────────
    # (pipeline imports asr_engine / translator which import faster_whisper /
    #  argostranslate — those must see OMP_NUM_THREADS from config.py)
    from pipeline import Pipeline  # noqa: PLC0415

    pipeline = Pipeline(source_lang=args.source, target_lang=args.target)

    pipeline.start()
    pipeline.wait()  # blocks until Ctrl+C or Enter
    sys.exit(0)  # explicit clean exit


if __name__ == "__main__":
    main()
