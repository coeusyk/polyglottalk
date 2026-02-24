"""
config.py — Global constants and environment variables for PolyglotTalk.

IMPORTANT: This module sets os.environ keys for CTranslate2 / OpenMP
BEFORE any faster_whisper or argostranslate imports happen anywhere
in the process. Import this module first in every entry point.
"""

import os

# ── Thread-count caps (must be set before importing CTranslate2 libs) ──────
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("CT2_INTER_THREADS", "1")

# ── Audio ───────────────────────────────────────────────────────────────────
SAMPLE_RATE: int = 16000          # Hz — Whisper expects 16 kHz
CHUNK_DURATION: float = 2.5       # seconds per ASR chunk
BLOCK_SIZE: int = int(SAMPLE_RATE * CHUNK_DURATION)  # 40 000 samples
# WSLg RDP audio bridge delivers lower amplitude than native Linux mics.
# Measured speech RMS ~0.0003; true silence ~0.00001. Threshold at 0.0001.
RMS_SILENCE_THRESHOLD: float = 0.0001  # chunks below this RMS are dropped

# ── Queue ───────────────────────────────────────────────────────────────────
QUEUE_MAXSIZE: int = 2            # backpressure limit per inter-stage queue
QUEUE_PUT_TIMEOUT: float = 1.0    # seconds before a blocked put retries
QUEUE_GET_TIMEOUT: float = 0.5    # seconds before a blocked get retries

# ── ASR (faster-whisper) ────────────────────────────────────────────────────
ASR_MODEL_SIZE: str = "base.en"
ASR_COMPUTE_TYPE: str = "int8"
ASR_DEVICE: str = "cpu"
ASR_BEAM_SIZE: int = 1
ASR_LANGUAGE: str = "en"          # skip language-detection for speed

# ── Translation (Argos Translate) ────────────────────────────────────────────
SOURCE_LANG: str = "en"
TARGET_LANG: str = "hi"
CONTEXT_MAXLEN: int = 2           # rolling source-segment window for prefix

# ── TTS (AI4Bharat IndicF5) ─────────────────────────────────────────────────
TTS_OUTPUT_DIR: str = "output"          # directory for saved TTS WAV files

INDICF5_MODEL_ID: str = "ai4bharat/IndicF5"

# Device for IndicF5 inference.  "auto" → cuda if available, else cpu.
# Set to "cpu" explicitly to force CPU-only mode.
INDICF5_DEVICE: str = "auto"

# Reference speech prompt used to clone voice/prosody characteristics.
# Downloaded by setup_models.py into the project's prompts/ directory.
INDICF5_REF_AUDIO_PATH: str = "prompts/HIN_F_HAPPY_00001.wav"

# Transcript of the reference audio.  Leave empty ("") to let IndicF5
# auto-transcribe it with Whisper on first use (slightly slower first call,
# then cached).
INDICF5_REF_TEXT: str = ""

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_FORMAT: str = "[%(asctime)s %(threadName)s] %(levelname)s %(message)s"
LOG_LEVEL: str = "INFO"

# ── ASR hallucination blocklist ──────────────────────────────────────────────
# faster-whisper commonly outputs these phrases on silence or near-silence.
# Comparisons are case-insensitive and strip punctuation/whitespace.
ASR_HALLUCINATION_BLOCKLIST: frozenset = frozenset({
    "thank you",
    "thanks",
    "thanks for watching",
    "thank you for watching",
    "you",
    "bye",
    "bye bye",
    "goodbye",
    "please subscribe",
    "like and subscribe",
    "see you next time",
    ".",
    "",
})
