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
