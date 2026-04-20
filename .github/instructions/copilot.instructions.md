# PolyglotTalk — GitHub Copilot Instructions

## Project Overview

PolyglotTalk is a **real-time, offline Speech-to-Speech Translation (S2ST) cascade pipeline** that runs on commodity hardware (WSL2, Ubuntu, NVIDIA RTX 4060 / CPU). It is a **research prototype**, not a production product. All design decisions are intentionally conservative: no new libraries without justification, no cloud dependencies, no changes to the threading architecture without explicit approval.

The pipeline translates English speech into a target Indian language in real time using three stages:

```
Microphone → [audio_queue] → ASREngine → [text_queue] → Translator → [tts_queue] → TTSEngine → WAV output
```

All four worker objects run on daemon threads. Models are loaded **once at startup**, never inside thread loops.

---

## Repository Layout

```
polyglot_talk/          # Core pipeline package
  config.py             # Single source of truth for ALL constants and model IDs
  pipeline.py           # Orchestrates 4-thread lifecycle (start / stop / drain)
  audio_capture.py      # sounddevice microphone capture thread
  asr_engine.py         # faster-whisper ASR + overlap deduplication
  translator.py         # Argos Translate MT + rolling context window
  tts_engine.py         # Facebook MMS-TTS synthesis → WAV files
  models.py             # Shared dataclasses: TextSegment, TranslatedSegment
benchmarks/             # Standalone measurement scripts (never import pipeline)
  benchmark_asr.py      # WER + latency on LibriSpeech dev-clean
  benchmark_mt.py       # BLEU + latency on sentences.txt
  benchmark_context.py  # Repetition / grammar break ablation
  benchmark_e2e.py      # Per-stage + total E2E latency over 20 trials
  system_meta.py        # Hardware snapshot sidecar (.meta.json per result)
tests/                  # pytest unit tests
results/                # CSV outputs from benchmarks (never committed by pipeline)
data/                   # Test sentences and reference data
```

---

## Architecture Rules (Non-Negotiable)

These constraints exist because the prototype was **measured and benchmarked against them**. Violating them invalidates paper results.

1. **Models are loaded once.** `ASREngine.__init__` and `Translator.__init__` load their models in the main thread before any thread is started. `TTSEngine.run()` loads its model on `TTSThread` (then signals `_model_ready`). `Pipeline.start()` waits on `_model_ready` before starting the other three threads. Never load a model inside a loop.

2. **Thread architecture is frozen.** Four threads, three `queue.Queue` objects, one `threading.Event` for shutdown. Do not add threads, remove threads, or change the queue topology without explicit instruction.

3. **config.py is the single source of truth.** All constants, model IDs, language codes, device strings, and tuning parameters live in `config.py`. Modules read `config.*` — they do not hardcode values. If a new constant is needed, add it to `config.py` first.

4. **No imports inside loops.** All `import` statements inside thread `run()` methods (e.g. `import torch`) are lazy one-time imports that happen before the processing loop, not inside it.

5. **Queue discipline: backpressure over blocking.** `Translator._put()` uses a drop-oldest strategy — if `tts_queue` is full, evict the oldest item before inserting the new one. Never block indefinitely on a `put()`.

6. **Shutdown via sentinels.** `None` is the sentinel value for every queue. `stop()` inserts one `None` per queue. Every `run()` loop checks `if item is None: break`.

---

## Language Code Namespaces

This is the most common source of silent bugs. There are **two different language code systems** in use and they must not be mixed:

| System | Format | Examples | Used by |
|---|---|---|---|
| ISO 639-1 | 2-letter | `en`, `hi`, `ta`, `te` | Argos Translate, `SOURCE_LANG`, `TARGET_LANG` in legacy code |
| ISO 639-3 | 3-letter | `hin`, `tam`, `tel`, `kan` | Facebook MMS-TTS model IDs |

`config.py` defines `ARGOS_LANG_MAP` to bridge from ISO 639-3 (`TARGET_LANG`) → ISO 639-1 for Argos. Always use this map — never hardcode a two-letter code where a three-letter code is expected or vice versa.

---

## config.py Conventions

- All constants are module-level, `UPPER_SNAKE_CASE`, with a type annotation.
- Every constant has an inline comment explaining **why** the value is what it is (not just what it is).
- The `MMS_TTS_MODEL_MAP` dict maps ISO 639-3 language codes → HuggingFace model IDs. To add a new TTS language, add one entry here. Nothing else needs to change.
- An `assert TARGET_LANG in MMS_TTS_MODEL_MAP` guard runs at import time. It must stay in place.
- `os.environ.setdefault("OMP_NUM_THREADS", "2")` must remain the **first** executable line — before any library imports that trigger CTranslate2 or OpenMP initialisation.

---

## Key Implementation Details

### ASR (asr_engine.py)
- Model: `faster-whisper` with `compute_type="int8"`, `beam_size=1` for minimum latency.
- **Overlap deduplication:** consecutive 2.5s chunks share 1.0s of audio. `deduplicateoverlap()` does normalised (lowercase, punctuation-stripped) suffix/prefix word matching to remove re-transcribed words. Do not weaken this — it prevents duplicate text from reaching the translator.
- **Sentence buffering:** ASR fragments accumulate until a natural sentence boundary or `SENTENCE_BUFFER_TIMEOUT` (5.0s). Do not flush on every chunk.
- **Hallucination blocklist:** `ASR_HALLUCINATION_BLOCKLIST` filters common Whisper silence hallucinations. Check this before adding repetition-handling elsewhere.

### MT (translator.py)
- Model: Argos Translate (`en → {target}`).
- **Rolling context window:** `_context_source` and `_context_translated` are `collections.deque(maxlen=CONTEXT_MAXLEN)`. The previous `CONTEXT_MAXLEN` source segments are prepended to each new input, and their cached translations are stripped from the output. **Do not remove or short-circuit this mechanism** — the context ablation benchmark (Issue #6) exists specifically because this is the primary contribution.
- `_translate_with_context()` makes **one** Argos call per segment, not two.
- `_trim_prefix()` does exact match first, difflib fuzzy fallback at 30% overlap threshold. The threshold is intentional — do not change it.
- The `_target_lang` passed to Argos must be ISO 639-1 (2-letter). Use `config.ARGOS_LANG_MAP[config.TARGET_LANG]` to resolve it.

### TTS (tts_engine.py)
- Model: `facebook/mms-tts-{lang}` (VITS-based, non-autoregressive). Loaded from `config.MMS_TTS_MODEL_MAP[config.TARGET_LANG]`.
- Output: WAV files at `config.TTS_OUTPUT_DIR/chunk_{id:04d}.wav`, not speaker playback (prevents mic feedback).
- Sample rate comes from `model.config.sampling_rate` — do not hardcode 16000 or 22050.
- `_model_ready` event must be `.set()` **after** the model is on the correct device and `.eval()` is called, and **before** the synthesis loop starts.

---

## Benchmarks

Benchmark scripts in `benchmarks/` are **standalone**. They do not import `pipeline.py` or any thread machinery. Rules:

- Models are loaded **once before the benchmark loop**, never inside it.
- Every script calls `system_meta.collect()` and writes a `.meta.json` sidecar alongside its CSV output.
- Result files go to `results/{stage}/`. Never overwrite existing result files — use a new filename when changing backends.
- Output CSV schema must not change for existing scripts (`benchmark_asr.py`, `benchmark_mt.py`, `benchmark_context.py`). New columns may be appended; existing columns may not be renamed or removed.
- `benchmark_e2e.py` saves to `results/e2e/e2e_latency_{backend_suffix}.csv`. Each backend swap gets its own file.

---

## Coding Standards

- **Python 3.11.9.** Do not use syntax or stdlib features introduced in 3.12+.
- **Type annotations on all public methods and module-level constants.** Use `from __future__ import annotations` at the top of every module.
- **`torch.no_grad()` as a context manager only** — not as a decorator.
- **No `print()` inside library code except for the live console `[→HI #NNN]` progress lines** in `translator.py` and `[TTS #NNN]` in `tts_engine.py`. These are intentional real-time feedback lines. Everything else uses `logger.*`.
- **`logging` over `print()` everywhere else.** Use `logger.debug` for per-chunk tracing, `logger.info` for lifecycle events, `logger.warning` for recoverable errors, `logger.exception` for caught exceptions.
- **Tests use `pytest` only** — no `unittest.TestCase`. Mock heavy models with `unittest.mock.patch` to keep tests fast and offline.
- **Every test has a one-line docstring** describing what it verifies.

---

## Dependency Policy

The following libraries are already installed and approved:

| Library | Purpose |
|---|---|---|
| `faster-whisper` | ASR |
| `transformers==4.49.0` | MMS-TTS (`VitsModel`, `VitsTokenizer`) |
| `argostranslate` | MT |
| `sounddevice`, `soundfile` | Audio I/O |
| `numpy`, `torch` (CUDA) | Tensor ops |
| `sacrebleu`, `jiwer` | Benchmark metrics |
| `datasets` | HuggingFace streaming datasets |

**Do not add new pip dependencies without explicit approval.** If a task can be solved with a library already in `requirements.txt`, use that library. If you believe a new dependency is needed, state the reason and the exact package name — do not add it silently.

`transformers` is pinned at **4.49.0**. Do not suggest upgrading it.

---

## Open Issues (as of develop branch)

| Issue | Title | Scope |
|---|---|---|
| #1 | `feat: language-aware ASR routing` | Input language / ASR model selection |
| #6 | `test: benchmark ASR candidates per language` | ASR benchmarking extension |
| #8 | `feat: multi-language TTS for Indian languages` | Output language / MMS-TTS model map |

When working on Issue #8, the only files that change are: `config.py` (add `MMS_TTS_MODEL_MAP`, `ARGOS_LANG_MAP`, update `TARGET_LANG`), `tts_engine.py` (resolve model ID from map), `translator.py` (resolve Argos target code from map), `setup_models.py` (iterate all language pairs), and `benchmark_e2e.py` (read model ID from map). No other files need changes.

---

## What NOT to Do

- Do not load models inside thread `run()` loops.
- Do not change `CONTEXT_MAXLEN`, the deduplication threshold (0.85), or the fuzzy prefix-trim threshold (0.30) without a benchmark justification.
- Do not replace `argostranslate` with MarianMT in the live pipeline — Argos outperforms MarianMT on both BLEU (0.46 vs 0.15) and latency (88ms vs 201ms) on this project's benchmark set. MarianMT exists only as a documented baseline in `benchmark_mt.py`.
- Do not add speaker playback to `tts_engine.py` — WAV file output is intentional to prevent microphone feedback.
- Do not use `localStorage` or anything browser-specific — this is a CLI Python application.
- Do not generate mock or fake benchmark results. All CSVs in `results/` must come from actual runs.
