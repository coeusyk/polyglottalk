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

# ── Overlapping chunks ──────────────────────────────────────────────────────
# Consecutive audio chunks share CHUNK_OVERLAP seconds of audio so that
# words at chunk boundaries are never cut.  The stride (new audio per chunk)
# is CHUNK_DURATION − CHUNK_OVERLAP.
#
# Research basis:
#   • Whispy (Bevilacqua et al., 2024) — shifting buffer with Levenshtein
#     deduplication achieves <2 % WER degradation vs offline Whisper.
#   • Whisper-Streaming (Machácek et al., 2023) — LocalAgreement-2 policy
#     with overlapping re-transcription achieves 3.3 s latency.
#   • Whisper long-form (OpenAI) — overlapping 30 s windows with timestamp-
#     based stitching avoid mid-word cuts.
CHUNK_OVERLAP: float = 1.0        # seconds of overlap between consecutive chunks
OVERLAP_SAMPLES: int = int(SAMPLE_RATE * CHUNK_OVERLAP)  # 16 000 samples (~2.5 words)
STRIDE_SAMPLES: int = BLOCK_SIZE - OVERLAP_SAMPLES        # 24 000 samples

# WSLg RDP audio bridge delivers lower amplitude than native Linux mics.
# Measured speech RMS ~0.0003; true silence ~0.00001. Threshold at 0.0001.
RMS_SILENCE_THRESHOLD: float = 0.0001  # chunks below this RMS are dropped

# ── Sentence accumulation ───────────────────────────────────────────────────
# ASR fragments are buffered until a natural sentence boundary is detected.
# This reduces the number of MT calls and produces better TTS prosody.
SENTENCE_BUFFER_TIMEOUT: float = 5.0   # flush after this many seconds of no new text
SENTENCE_BUFFER_MAXWORDS: int = 25     # force-flush when buffer exceeds this many words

# ── Queue ───────────────────────────────────────────────────────────────────
QUEUE_MAXSIZE: int = 2            # backpressure limit per inter-stage queue
QUEUE_PUT_TIMEOUT: float = 1.0    # seconds before a blocked put retries
QUEUE_GET_TIMEOUT: float = 0.5    # seconds before a blocked get retries

# ── ASR (faster-whisper) ────────────────────────────────────────────────────
ASR_MODEL_SIZE: str = "base.en"
ASR_COMPUTE_TYPE: str = "int8"
ASR_DEVICE: str = "auto"
ASR_BEAM_SIZE: int = 1
ASR_LANGUAGE: str = "en"          # skip language-detection for speed

ASR_STRIP_TRAILING_PERIOD: bool = True

# ── Language codes — IMPORTANT: two different namespaces are in use ─────────
# MMS-TTS uses ISO 639-3 (three-letter):  "hin", "tam", "tel", "kan", "ben", …
# Argos Translate uses ISO 639-1 (two-letter): "hi" only (for Indian langs)
# MarianMT (Helsinki-NLP) uses ISO 639-1 (two-letter): "ta", "te", "kn", …
# TARGET_LANG is always the ISO 639-3 key used by MMS-TTS.
# SOURCE_LANG stays ISO 639-1 ("en") because Argos, MarianMT, and Whisper
# all use it for the source side.

# ── Translation backend routing ─────────────────────────────────────────────
SOURCE_LANG: str = "en"       # ISO 639-1 — shared by Whisper ASR and all MT backends

# Active output language.  Must be a key in both MMS_TTS_MODEL_MAP and
# ARGOS_LANG_MAP or MARIANMT_MODEL_MAP.  Changing only this constant
# switches the full pipeline (ASR → MT → TTS).
TARGET_LANG: str = "hin"      # ISO 639-3 — used as primary language key

# Languages for which Argos Translate publishes an en→xx offline package.
# As of 2025, only Hindi is available for Indian languages via argospm.
# All others fall through to MarianMT.
ARGOS_SUPPORTED_LANGS: frozenset[str] = frozenset({"hin"})

# ISO 639-3 → ISO 639-1 bridge for Argos Translate (Hindi only).
ARGOS_LANG_MAP: dict[str, str] = {
    "hin": "hi",   # Hindi — only Indian language with an Argos en→xx package
}

# ISO 639-3 → HuggingFace MarianMT checkpoint for all non-Hindi Indian langs.
# Helsinki-NLP opus-mt models are offline, ~300 MB each, load via transformers.
MARIANMT_MODEL_MAP: dict[str, str] = {
    "tam": "Helsinki-NLP/opus-mt-en-ta",   # Tamil
    "tel": "Helsinki-NLP/opus-mt-en-te",   # Telugu
    "kan": "Helsinki-NLP/opus-mt-en-kn",   # Kannada
    "ben": "Helsinki-NLP/opus-mt-en-bn",   # Bengali
    "mal": "Helsinki-NLP/opus-mt-en-ml",   # Malayalam
    "mar": "Helsinki-NLP/opus-mt-en-mr",   # Marathi
    "guj": "Helsinki-NLP/opus-mt-en-gu",   # Gujarati
}

# MT_BACKEND is derived automatically from TARGET_LANG — do not set manually.
# Values: "argos" | "marian"
MT_BACKEND: str = "argos" if TARGET_LANG in ARGOS_SUPPORTED_LANGS else "marian"

CONTEXT_MAXLEN: int = 2           # rolling source-segment window for prefix

# ── TTS (Facebook MMS-TTS, VITS-based) ──────────────────────────────────────
TTS_OUTPUT_DIR: str = "output"          # directory for saved TTS WAV files

# MMS-TTS model routing: maps TARGET_LANG (ISO 639-3) → HuggingFace checkpoint.
# Each language has its own VITS weights; all use the same VitsModel interface.
# To add a new language: add one entry here AND one entry in either
# ARGOS_LANG_MAP (if Argos supports it) or MARIANMT_MODEL_MAP (otherwise).
MMS_TTS_MODEL_MAP: dict[str, str] = {
    "hin": "facebook/mms-tts-hin",   # Hindi
    "tam": "facebook/mms-tts-tam",   # Tamil
    "tel": "facebook/mms-tts-tel",   # Telugu
    "kan": "facebook/mms-tts-kan",   # Kannada
    "ben": "facebook/mms-tts-ben",   # Bengali
    "mal": "facebook/mms-tts-mal",   # Malayalam
    "mar": "facebook/mms-tts-mar",   # Marathi
    "guj": "facebook/mms-tts-guj",   # Gujarati
}

# Validate TARGET_LANG at import time — fail fast rather than deep in a thread.
assert TARGET_LANG in MMS_TTS_MODEL_MAP, (
    f"TARGET_LANG={TARGET_LANG!r} has no MMS-TTS checkpoint. "
    f"Valid values: {sorted(MMS_TTS_MODEL_MAP)}"
)

# Every language must have either an Argos or a MarianMT entry — never neither.
_ALL_MT_LANGS = set(ARGOS_LANG_MAP) | set(MARIANMT_MODEL_MAP)
assert set(MMS_TTS_MODEL_MAP).issubset(_ALL_MT_LANGS), (
    f"These TTS languages have no MT backend: "
    f"{set(MMS_TTS_MODEL_MAP) - _ALL_MT_LANGS}"
)

# Device for MMS-TTS inference.  "auto" → cuda if available, else cpu.
MMS_TTS_DEVICE: str = "auto"

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_FORMAT: str = "[%(asctime)s %(threadName)s] %(levelname)s %(message)s"
LOG_LEVEL: str = "INFO"

# ── ASR hallucination blocklist ──────────────────────────────────────────────
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
