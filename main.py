"""
main.py — Entry point for PolyglotTalk.

Usage
-----
    python main.py                          # CLI only, default language
    python main.py --source en --target hin
    python main.py --dashboard              # dashboard-only, start from UI
    python main.py --dashboard --target hin # dashboard + auto-start pipeline
    python main.py --log-level DEBUG

IMPORTANT: config is imported first so that OMP_NUM_THREADS and
CT2_INTER_THREADS are set in os.environ before any CTranslate2 library
(faster-whisper, argostranslate) is imported anywhere in the process.
"""

from __future__ import annotations

# ── config MUST be the first project import ──────────────────────────────────
from polyglot_talk import config  # noqa: F401 — sets os.environ before CT2 is imported

import argparse
import logging
import sys
import time
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
        default=None,          # None = let the dashboard UI decide
        metavar="LANG",
        help="Target language ISO 639-3 code. If omitted in --dashboard mode, "
             "the pipeline is started from the UI.",
    )
    p.add_argument(
        "--log-level",
        default=config.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    p.add_argument(
        "--dashboard",
        action="store_true",
        default=False,
        help="Start the real-time WebSocket dashboard server",
    )
    p.add_argument(
        "--dashboard-port",
        type=int,
        default=8765,
        metavar="PORT",
        help="Port for the dashboard WebSocket server (default: 8765)",
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
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)

    # ── Resolve device label for banner ───────────────────────────────────
    device = config.MMS_TTS_DEVICE
    if device == "auto":
        try:
            import torch  # noqa: PLC0415
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    target_display = (args.target or config.TARGET_LANG).upper()
    print("=" * 60)
    print(" PolyglotTalk v0.1 — Offline Speech-to-Speech Translation")
    print(f" {args.source.upper()} → {target_display}  |  TTS: MMS-TTS ({device})  |  No cloud APIs")
    print(f" TTS output saved to: {config.TTS_OUTPUT_DIR}/chunk_NNNN.wav")
    print("=" * 60)

    # ── Clean up old output chunks ────────────────────────────────────────
    output_dir = Path(config.TTS_OUTPUT_DIR)
    if output_dir.exists():
        for wav_file in output_dir.glob("chunk_*.wav"):
            wav_file.unlink()
            logging.getLogger(__name__).debug("Removed old chunk: %s", wav_file.name)

    # ══════════════════════════════════════════════════════════════════════
    # DASHBOARD MODE
    # ══════════════════════════════════════════════════════════════════════
    if args.dashboard:
        import threading
        from dashboard_server import run_server, pipeline_manager

        # Start FastAPI / WebSocket server in a daemon thread
        dash_thread = threading.Thread(
            target=run_server,
            kwargs={"host": "0.0.0.0", "port": args.dashboard_port},
            name="DashboardServer",
            daemon=True,
        )
        dash_thread.start()
        time.sleep(1.0)  # give uvicorn a moment to bind

        print(f"  Dashboard: http://localhost:{args.dashboard_port}  "
              f"(WS: ws://localhost:{args.dashboard_port}/ws)")

        if args.target:
            # Auto-start the pipeline with the CLI-specified language
            print(f"  Auto-starting pipeline: {args.source} → {args.target}")
            pipeline_manager.start(source_lang=args.source, target_lang=args.target)
        else:
            print("  Open the dashboard and press ▶ Start to begin.")

        # Keep main thread alive; pipeline is controlled entirely from the UI
        print("  Press Ctrl+C to shut down.")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nInterrupt received — shutting down…")
            pipeline_manager.stop()
            time.sleep(1.0)  # let stop() propagate
        sys.exit(0)

    # ══════════════════════════════════════════════════════════════════════
    # CLI-ONLY MODE (no dashboard)
    # ══════════════════════════════════════════════════════════════════════
    target = args.target or config.TARGET_LANG

    from polyglot_talk.pipeline import Pipeline  # noqa: PLC0415

    pipeline = Pipeline(source_lang=args.source, target_lang=target)
    pipeline.start()
    pipeline.wait()  # blocks until Ctrl+C or Enter
    sys.exit(0)


if __name__ == "__main__":
    main()
