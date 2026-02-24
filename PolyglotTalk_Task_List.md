# PolyglotTalk — PCL Project Task List
> **Goal**: Build a working prototype of a real-time, offline Speech-to-Speech Translation system and document it as a research paper that showcases the concept, design decisions, and experimental findings.

---

## Part 1 — Prototype (Make It Work)

### 1.1 Stability Fixes
- [ ] Upgrade faster-whisper model to `base.en` in `config.py` if not already done (`ASR_MODEL_SIZE = "base.en"`)
- [ ] Add RMS energy check in `asr_engine.py` before pushing to `text_queue` — skip silent chunks where `np.sqrt(np.mean(audio**2)) < 0.01`
- [ ] Add duplicate transcription guard — if `new_text == previous_text`, skip and do not push to `text_queue`
- [ ] Confirm faster-whisper generator is fully drained: `" ".join(seg.text for seg in segments)` — never break early from the loop
- [ ] Set `OMP_NUM_THREADS=2` and `CT2_INTER_THREADS=1` as environment variables in `config.py` before any imports

### 1.2 Context Continuity
- [ ] Confirm `_translate_with_context()` correctly builds `combined_input = f"{prefix_source} {new_text}".strip()`
- [ ] Add exact prefix trim: if `full_translation.startswith(prefix_translated)`, strip it
- [ ] Add `difflib.SequenceMatcher` fuzzy trim as fallback when exact trim fails
- [ ] Add safety guard: never return an empty string — fall back to `full_translation` as-is if trimmed result is empty

### 1.3 Threading & Shutdown
- [ ] Confirm `pyttsx3.init()` is the **first line** inside `TTSEngine.run()` — not in `__init__()`
- [ ] Replace blocking `queue.put()` calls with `put(item, timeout=1.0)` inside a loop that checks `stop_event`
- [ ] Confirm all 4 threads are set as daemon threads — Ctrl+C must exit cleanly without hanging

### 1.4 Latency Logging *(required for experiments)*
- [ ] Replace raw `str`/`ndarray` in queues with `AudioChunk`, `TextSegment`, `TranslatedSegment` dataclasses, each carrying `chunk_id: int` and `timestamp: float`
- [ ] Log `time.perf_counter()` at entry and exit of `_transcribe()`, `_translate_with_context()`, and `_speak()`
- [ ] Log per-chunk end-to-end latency: `[pipeline] Chunk #N E2E latency: X.XXs`
- [ ] Print live console output on each processed chunk:
  ```
  [ASR  #1] Hello, how are you today?
  [→HI  #1] नमस्ते, आप आज कैसे हैं?
  ```

### 1.5 Component Tests *(verify before running experiments)*
- [ ] `test_audio_capture.py` — record 3s from mic, assert WAV file size > 0 and RMS > 0.001
- [ ] `test_asr.py` — transcribe a known `hello.wav` clip, assert "hello" appears in output
- [ ] `test_translator.py` — translate "Hello, how are you?" en→hi, assert Devanagari characters in output
- [ ] `test_tts.py` — speak "Testing one two three" inside a child thread, assert no exception
- [ ] `test_context.py` — unit test empty context, 1-segment, 2-segment, fuzzy trim, empty input skip, repeated input skip
- [ ] `test_pipeline_e2e.py` — feed 3 WAV chunks via mock AudioCapture, assert 2+ translated outputs within 15s, assert all threads shut down within 5s

---

## Part 2 — Experiments *(data for the paper)*

### 2.1 Prepare Test Data
- [ ] Download 20–30 Common Voice English clips (2.5s each) from `cv-corpus` delta segment
- [ ] Create `test_clips/ground_truth.txt` — one correct transcription per clip for WER calculation
- [ ] Create `test_sentences/sentences.txt` — 20–30 English sentences of varied length and complexity for MT benchmarking

### 2.2 Experiment 1 — ASR Comparison
> *Which faster-whisper model size best balances accuracy and speed?*

- [ ] Write `benchmarks/benchmark_asr.py` — runs each model on all test clips, records WER and latency per clip, saves to `results/asr_results.csv`
- [ ] Run **faster-whisper `tiny.en`** — record WER + average latency
- [ ] Run **faster-whisper `base.en`** — record WER + average latency *(primary model)*
- [ ] Run **faster-whisper `small.en`** — record WER + average latency
- [ ] Populate ASR results table in paper (Section 5)

### 2.3 Experiment 2 — MT Comparison
> *Does Argos Translate or MarianMT produce better translations at lower latency?*

- [ ] Write `benchmarks/benchmark_mt.py` — runs each model on test sentences, records BLEU score and latency per sentence, saves to `results/mt_results.csv`
- [ ] Run **Argos Translate** en→hi — record BLEU + average latency *(primary model)*
- [ ] Run **MarianMT** (`Helsinki-NLP/opus-mt-en-hi`, `num_beams=1`) — record BLEU + average latency
- [ ] Populate MT results table in paper (Section 5)

### 2.4 Experiment 3 — End-to-End Pipeline Latency
> *What is the actual total delay from speaking to hearing the translation?*

- [ ] Run the full pipeline for 20 trials using the latency logs from Part 1.4
- [ ] Record per-stage latency for each trial: ASR time, MT time, TTS time, total E2E time
- [ ] Calculate mean and standard deviation for each stage
- [ ] Save to `results/e2e_latency.csv`
- [ ] Populate E2E latency table in paper (Section 5)

### 2.5 Experiment 4 — Context Continuity Validation
> *Does the context window actually reduce repetitions and grammar breaks?*

- [ ] Prepare a 10-sentence scripted conversation in `test_clips/conversation_script.txt`
- [ ] Run pipeline on the conversation **with** context window — manually count repetitions and grammar breaks across all outputs
- [ ] Run pipeline **without** context window (disable prefix injection in `_translate_with_context()`) — manually count repetitions and grammar breaks
- [ ] Record the difference as a table: `{repetitions_with, repetitions_without, grammar_breaks_with, grammar_breaks_without}`
- [ ] Populate context continuity results in paper (Section 5) — this is your primary contribution result

---

## Part 3 — Research Paper

> **Title**: *PolyglotTalk: A Real-Time Offline Speech-to-Speech Translation System Using a Context-Aware Cascade Pipeline*
>
> **Core argument**: A CPU-only, fully offline cascade S2ST pipeline with a rolling context window is a practical and effective approach for multilingual communication on low-resource devices, validated through component-level and end-to-end benchmarking.

### 3.1 Sections — Write in This Order

#### Step 1 — System Design *(write first, no results needed)*
- [ ] Describe the 4-thread pipeline: AudioCapture → ASR → Translator → TTS
- [ ] Include the pipeline architecture diagram (use the project flowchart)
- [ ] Include the threading timing diagram (from the implementation plan) showing pipelined parallel execution
- [ ] Write **Algorithm 1 — Context-Aware Translation**:
  ```
  Input:  text_chunk, context_window (deque, maxlen=2)
  1. prefix ← join(context_window)
  2. combined ← concat(prefix, text_chunk)
  3. full_translation ← MT_model(combined)
  4. trimmed ← remove_prefix(full_translation, translated(prefix))
  5. context_window.append(text_chunk)
  6. return trimmed
  ```
- [ ] Describe the drop-oldest backpressure strategy and why it keeps the pipeline real-time

#### Step 2 — Related Work
- [ ] Describe cascade S2ST (ASR→MT→TTS) and its known limitations (error propagation, no prosody)
- [ ] Describe direct/end-to-end S2ST (Translatotron, Google 2025) and its known limitations (GPU requirement, data scarcity)
- [ ] State the gap: *no published work benchmarks open-source cascade components together under a CPU-only, offline deployment constraint*
- [ ] Cite 6–8 papers: Translatotron 1/2, CrossVoice (2024), Google S2ST blog (2025), IWSLT 2024, Argos Translate, faster-whisper

#### Step 3 — Experiments & Results *(write after Part 2 data is collected)*
- [ ] Describe hardware used (CPU specs, RAM, OS)
- [ ] Describe test dataset (Common Voice, number of clips, duration)
- [ ] Describe evaluation metrics: WER, BLEU, E2E latency (mean ± std), repetition count
- [ ] Insert ASR comparison table (Experiment 1)
- [ ] Insert MT comparison table (Experiment 2)
- [ ] Insert E2E latency table (Experiment 3)
- [ ] Insert context continuity comparison table (Experiment 4)

#### Step 4 — Discussion
- [ ] State which ASR model you recommend and why (accuracy vs latency tradeoff)
- [ ] State which MT model you recommend and why
- [ ] Quantify the improvement from context continuity — this is your key finding
- [ ] Honestly state limitations: no voice cloning, fixed 2.5s buffer, pyttsx3 robotic voice
- [ ] Note that better Indian language support (e.g., Sarvam Translate) is a direct upgrade path

#### Step 5 — Introduction *(write after results are known)*
- [ ] Open with the real-world problem: multilingual India, unreliable connectivity, no affordable offline calling translation
- [ ] State what existing solutions miss: Google Meet needs internet, Pixel 10 needs expensive hardware
- [ ] State what your system does differently: offline, CPU-only, any device, modular
- [ ] End with paper structure: "Section 2 covers related work, Section 3 describes system design..."

#### Step 6 — Abstract & Conclusion *(write last)*
- [ ] **Abstract** (150 words): Problem → approach → key result → implication
- [ ] **Conclusion** (0.5 page): What you built, what you found, what comes next (mobile calling integration)

### 3.2 Paper Checklist Before Submission
- [ ] All 4 experiment result tables are populated with real numbers
- [ ] Architecture diagram is clean and legible (export flowchart as high-res PNG)
- [ ] Algorithm 1 is formatted correctly
- [ ] All citations are in consistent format (IEEE or ACM — check what your institution requires)
- [ ] Paper is 7–8 pages (not more, not less)
- [ ] Mentor has reviewed and given feedback
- [ ] Final proofread complete

---

## Recommended Timeline

| Week | Focus |
|------|-------|
| **Week 1** | Part 1.1–1.3 — Fix all stability and threading bugs |
| **Week 2** | Part 1.4–1.5 — Add latency logging, run all component tests |
| **Week 3** | Part 2.1–2.5 — Collect test data, run all 4 experiments |
| **Week 4** | Part 3, Steps 1–2 — Write System Design and Related Work |
| **Week 5** | Part 3, Steps 3–4 — Write Results and Discussion |
| **Week 6** | Part 3, Steps 5–6 — Write Introduction and Abstract, mentor review, finalize |
