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
# This ensures the translator and TTS receive complete sentences rather than
# mid-word fragments with artificial trailing periods.
#
# SENTENCE_BUFFER_TIMEOUT must be larger than the natural gap between consecutive
# ASR text outputs on CPU:
#   gap = STRIDE_SAMPLES/SAMPLE_RATE + whisper_transcription_time
#       = (CHUNK_DURATION - CHUNK_OVERLAP) + CHUNK_DURATION * ~0.8
#       ≈ 1.5s + 2.0s = 3.5s on CPU base.en int8
# We add a generous safety margin so brief speech pauses do NOT cause premature
# flushes.  The silence-based flush (RMS filter) handles genuine sentence ends.
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

# Strip trailing periods that Whisper auto-appends to every chunk.
# This prevents the translator / TTS from treating every fragment as a
# complete sentence, which degrades translation quality and prosody.
ASR_STRIP_TRAILING_PERIOD: bool = True

# ── Language codes — IMPORTANT: two different namespaces are in use ────────────
# MMS-TTS uses ISO 639-3 (three-letter):  "hin", "tam", "tel", "kan", "ben", …
# Argos Translate uses ISO 639-1 (two-letter): "hi", "ta", "te", "kn", "bn", …
# These are DIFFERENT and must NOT be mixed.  ARGOS_LANG_MAP bridges them.
# TARGET_LANG is always the ISO 639-3 key used by MMS-TTS.
# SOURCE_LANG stays ISO 639-1 ("en") because Argos and Whisper both use it.

# ── Translation (Argos Translate) ────────────────────────────────────────────
SOURCE_LANG: str = "en"       # ISO 639-1 — shared by Whisper ASR and Argos MT

# Active output language.  Must be a key in both MMS_TTS_MODEL_MAP and
# ARGOS_LANG_MAP.  Changing only this constant switches the full pipeline.
TARGET_LANG: str = "hin"           # ISO 639-3 — used by MMS-TTS model IDs

# ISO 639-3 → ISO 639-1 bridge for Argos Translate.
# Argos uses two-letter codes; MMS-TTS uses three-letter codes.
# Add a new entry here whenever a new language is added to MMS_TTS_MODEL_MAP.
ARGOS_LANG_MAP: dict[str, str] = {
    "hin": "hi",   # Hindi
    "tam": "ta",   # Tamil
    "tel": "te",   # Telugu
    "kan": "kn",   # Kannada
    "ben": "bn",   # Bengali
    "mal": "ml",   # Malayalam
    "mar": "mr",   # Marathi
    "guj": "gu",   # Gujarati
}

CONTEXT_MAXLEN: int = 2           # rolling source-segment window for prefix

# ── TTS (Facebook MMS-TTS, VITS-based) ──────────────────────────────────────
TTS_OUTPUT_DIR: str = "output"          # directory for saved TTS WAV files

# MMS-TTS model routing: maps TARGET_LANG (ISO 639-3) → HuggingFace checkpoint.
# Each language has its own VITS weights; all use the same VitsModel interface.
# To add a new language: add one entry here AND one entry in ARGOS_LANG_MAP.
# No other file needs to change.
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

# Device for MMS-TTS inference.  "auto" → cuda if available, else cpu.
# Set to "cpu" explicitly to force CPU-only mode.
MMS_TTS_DEVICE: str = "auto"

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
