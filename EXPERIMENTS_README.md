# Part 2 — Experiments

Benchmarking suite for the PolyglotTalk pipeline. These experiments measure component-level and end-to-end performance to populate the results tables in the research paper (Section 5).

---

## Quick Start

```bash
# 1. Generate test audio clips (synthesised from ground truth text)
python test_clips/generate_test_clips.py

# 2. Run all four experiments
python benchmarks/benchmark_asr.py       # Experiment 1 — ASR comparison
python benchmarks/benchmark_mt.py        # Experiment 2 — MT comparison
python benchmarks/benchmark_e2e.py       # Experiment 3 — E2E latency
python benchmarks/benchmark_context.py   # Experiment 4 — Context continuity
```

All results are saved as CSVs in the `results/` directory.

---

## Test Data

| File | Description |
|------|-------------|
| `test_clips/ground_truth.txt` | 25 sentences with their exact transcriptions for WER |
| `test_clips/clip_01.wav` … `clip_25.wav` | Synthesised WAV clips (16 kHz mono, ~2.5 s each) |
| `test_sentences/sentences.txt` | 25 English → Hindi sentence pairs for BLEU scoring |
| `test_clips/conversation_script.txt` | 10-sentence scripted dialogue for context testing |
| `test_clips/generate_test_clips.py` | Script that synthesises clips via pyttsx3 |

Clips are generated with `pyttsx3` so that ground truth transcriptions are exact — no manual labelling required.

---

## Experiment 1 — ASR Model Comparison

**Script:** `benchmarks/benchmark_asr.py`
**Question:** *Which faster-whisper model size best balances accuracy and speed?*

Benchmarks three model sizes (`tiny.en`, `base.en`, `small.en`) on all 25 test clips. For each clip the script records:

- **WER** (Word Error Rate) — Levenshtein edit distance at word level
- **Latency** — wall-clock inference time

### Results

| Model | Avg WER | Avg Latency | Std Latency |
|---------|---------|-------------|-------------|
| tiny.en | 19.64 % | 0.405 s | 0.024 s |
| **base.en** | **18.18 %** | **0.826 s** | **0.015 s** |
| small.en | 17.24 % | 2.604 s | 0.047 s |

**Recommendation:** `base.en` — only 1.4 % worse WER than `small.en` but **3× faster**. `tiny.en` is fastest but sacrifices accuracy.

**Output files:** `results/asr_results.csv` (per-clip detail), `results/asr_summary.csv`

---

## Experiment 2 — MT Model Comparison

**Script:** `benchmarks/benchmark_mt.py`
**Question:** *Does Argos Translate or MarianMT produce better translations at lower latency?*

Translates all 25 test sentences (en → hi) with both engines and scores each against the Hindi reference.

- **BLEU** — sentence-level BLEU via `sacrebleu` (normalised to 0–1)
- **Latency** — wall-clock time per sentence

### Results

| Model | Avg BLEU | Avg Latency | Std Latency |
|-------|----------|-------------|-------------|
| **Argos Translate** | **0.523** | **0.162 s** | 0.361 s |
| MarianMT (opus-mt-en-hi) | 0.271 | 0.301 s | 0.128 s |

**Recommendation:** Argos Translate achieves ~2× higher BLEU at half the latency and requires no GPU.

**Output files:** `results/mt_results.csv` (per-sentence detail), `results/mt_summary.csv`

---

## Experiment 3 — End-to-End Pipeline Latency

**Script:** `benchmarks/benchmark_e2e.py`
**Question:** *What is the actual total delay from speaking to hearing the translation?*

Runs 20 trials through the full **ASR → MT → TTS** pipeline (single-threaded, deterministic timing). Each trial feeds one test clip and records per-stage wall-clock time.

### Results

| Stage | Mean | Std |
|-------|------|-----|
| ASR | 0.762 s | 0.037 s |
| MT | 0.119 s | 0.261 s |
| TTS | 0.318 s | 0.382 s |
| **Total E2E** | **1.199 s** | **0.527 s** |

ASR dominates the pipeline at ~64 % of total latency. MT is the cheapest stage.

**Output file:** `results/e2e_latency.csv` (20 trial rows + mean/std summary)

---

## Experiment 4 — Context Continuity Validation

**Script:** `benchmarks/benchmark_context.py`
**Question:** *Does the rolling context window reduce repetitions and grammar breaks?*

Feeds a 10-sentence scripted conversation through the Translator in two conditions:

1. **With context** — uses the rolling 2-segment prefix (same as the live pipeline)
2. **Without context** — translates each sentence independently

Automatically detects:

- **Repetitions** — consecutive outputs sharing > 60 % of their words
- **Grammar breaks** — outputs that are too short (< 3 chars) or identical to the English input

### Results

| Metric | With Context | Without Context |
|--------|-------------|-----------------|
| Repetitions | 0 | 0 |
| Grammar Breaks | 0 | 0 |
| Avg Latency | 0.231 s | 0.084 s |

Both conditions showed 0 issues on this clean scripted dialogue, indicating Argos Translate is robust on short structured sentences. The context window adds ~0.15 s of overhead per sentence due to the longer combined input. For longer, more ambiguous, or noisy real-world conversations the context window is expected to show more pronounced benefits.

**Output files:** `results/context_results.csv` (summary), `results/context_detail.csv` (per-sentence)

---

## Dependencies

All experiment dependencies are listed in `requirements.txt`:

| Package | Purpose |
|---------|---------|
| `faster-whisper` | ASR engine (Experiments 1, 3) |
| `argostranslate` | Primary MT engine (Experiments 2, 3, 4) |
| `pyttsx3` | TTS engine + clip generation (Experiment 3) |
| `sacrebleu` | BLEU scoring (Experiment 2) |
| `transformers` + `sentencepiece` + `torch` | MarianMT model (Experiment 2) |
| `numpy`, `scipy` | Audio I/O and signal processing |

---

## Directory Structure

```
polyglot-talk/
├── benchmarks/
│   ├── __init__.py
│   ├── benchmark_asr.py          # Experiment 1
│   ├── benchmark_mt.py           # Experiment 2
│   ├── benchmark_e2e.py          # Experiment 3
│   └── benchmark_context.py      # Experiment 4
├── test_clips/
│   ├── generate_test_clips.py    # Synthesises WAV clips
│   ├── ground_truth.txt          # Clip → transcription mapping
│   ├── conversation_script.txt   # 10-sentence dialogue
│   └── clip_01.wav … clip_25.wav # Generated audio
├── test_sentences/
│   └── sentences.txt             # EN|HI sentence pairs
└── results/
    ├── asr_results.csv           # Per-clip ASR data
    ├── asr_summary.csv           # Model-level ASR summary
    ├── mt_results.csv            # Per-sentence MT data
    ├── mt_summary.csv            # Model-level MT summary
    ├── e2e_latency.csv           # 20-trial E2E timing
    ├── context_results.csv       # With/without comparison
    └── context_detail.csv        # Per-sentence context detail
```
