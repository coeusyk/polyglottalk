"""
main.py — Entry point for PolyglotTalk.

Usage
-----
    python main.py
    python main.py --source en --target hi
    python main.py --source en --target hi --tts-rate 160

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
        "--tts-rate",
        type=int,
        default=config.TTS_RATE,
        metavar="WPM",
        help=f"TTS speech rate in words-per-minute (default: {config.TTS_RATE})",
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

    print("=" * 60)
    print(" PolyglotTalk v0.1 — Offline Speech-to-Speech Translation")
    print(f" {args.source.upper()} → {args.target.upper()}  |  CPU-only  |  No cloud APIs")
    print("=" * 60)

    # ── Import pipeline here so CT2 env vars are already set ─────────────
    # (pipeline imports asr_engine / translator which import faster_whisper /
    #  argostranslate — those must see OMP_NUM_THREADS from config.py)
    from pipeline import Pipeline  # noqa: PLC0415

    pipeline = Pipeline(source_lang=args.source, target_lang=args.target)

    # Apply CLI override for TTS rate if provided
    if args.tts_rate != config.TTS_RATE:
        pipeline._tts_engine._rate = args.tts_rate

    pipeline.start()
    pipeline.wait()  # blocks until Ctrl+C


if __name__ == "__main__":
    main()
