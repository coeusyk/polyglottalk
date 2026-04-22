# PHASES.md — PolyglotTalk Development Roadmap

> **Project:** Offline Speech-to-Speech Translation (S2ST)  
> **Target**: IEEE Access journal submission    
> **Long-term vision**: any-to-any language S2ST, fully offline 

---

## Phase 0 — Foundation (COMPLETE)
**Goal:** Working EN→HIN cascade pipeline, offline, no cloud APIs.

### Delivered
- [x] faster-whisper ASR (base.en, int8) with overlapping chunk capture
- [x] Argos Translate MT (en→hin)
- [x] MMS-TTS (`facebook/mms-tts-hin`) on CUDA
- [x] Sentence buffer with basic prefix/suffix deduplication
- [x] WebSocket dashboard server with real-time transcript display
- [x] Output WAV saving (`output/chunk_NNNN.wav`)
- [x] Pipeline start/stop lifecycle management

### Known Debt Carried Forward
- Sentence buffer is append-only; semantic restating causes double-speech (see Phase 1 fix)
- Near-duplicate guard operates on full raw chunk text, not deduped text
- No word-level timestamps used despite faster-whisper supporting them

---

## Phase 1 — ASR Quality & Overlap Fix (CURRENT)
**Goal:** Eliminate recurring double-speech artifacts from overlapping chunk boundaries.
**Branch:** `multi-language-tts` (merge → develop when complete)

### Tasks
- [ ] Implement tail-replacement logic in `ASREngine.run`:
  - Compare new chunk against last N words of `_sentence_buf`
  - If Jaccard overlap > 0.6 and new chunk is a plausible restatement, replace tail instead of appending
  - Log replacements at DEBUG level for observability
- [ ] Relax or remove full-chunk near-duplicate guard (currently 85% threshold on raw text)
- [ ] Add `_word_overlap_ratio(a, b)` helper to `asr_engine.py`
- [ ] Write regression test:
  - Input: two consecutive chunks where chunk B restates chunk A's tail differently
  - Assert: final `SENT` string contains only one version of the phrase
- [ ] Write unit test for `_deduplicate_overlap` edge cases (empty input, single word, identical chunks)

### Acceptance Criteria
- Running the EN→HIN pipeline for 2 minutes of continuous speech produces no audible double-speech in TTS output
- All new tests pass in CI

---

## Phase 2 — Multi-Language Support (Indic-first)
**Goal:** Extend pipeline to support additional Indic language pairs for the second language in the journal evaluation.
**Minimum required for IEEE Access submission:** at least one additional language pair beyond EN→HIN.

### Language Pairs Planned
| Pair | ASR | MT | TTS | Priority |
|---|---|---|---|---|
| EN → BEN (Bengali) | faster-whisper | Argos / MarianMT | MMS-TTS hin→ben | High |
| EN → TAM (Tamil) | faster-whisper | Argos / MarianMT | MMS-TTS tam | High |
| EN → TEL (Telugu) | faster-whisper | Argos | MMS-TTS tel | Medium |
| HIN → BEN | faster-whisper (multilingual) | MarianMT | MMS-TTS ben | Medium |

### Tasks
- [ ] Abstract `TTSEngine` to accept language code at runtime (not hardcoded `facebook/mms-tts-hin`)
- [ ] Abstract `TranslatorEngine` to accept source/target language pair at runtime
- [ ] Test Argos package availability for each planned language pair
- [ ] Validate MMS-TTS model quality for BEN and TAM (listen test + RTF measurement)
- [ ] Update CLI and dashboard to expose language pair selector
- [ ] Run at least one full end-to-end test per new language pair

### Note
Any-to-any language support (e.g., non-Indic pairs, low-resource languages beyond Indic) is long-term future work and will not be in scope for the journal submission. It will be described in the Future Work section of the paper.

---

## Phase 3 — Benchmarking & Evaluation (Journal-critical)
**Goal:** Produce the controlled experiment results required for IEEE Access. Every claim in the paper must trace to a row in these tables.

### Experiment Suite

#### 3A — ASR Model Comparison
- Models: `base.en`, `small.en`, `medium.en` (int8, faster-whisper)
- Metric: WER on a fixed Hindi-accented English test set (minimum 50 utterances, ~5 min audio)
- Controlled variable: all other pipeline components held constant
- [ ] Collect or curate test audio corpus (Hindi-accented English speakers)
- [ ] Run WER evaluation script for each model
- [ ] Record: WER (%), model load time (s), inference RTF per chunk

#### 3B — MT Engine Comparison
- Engines: Argos Translate vs MarianMT (same language pair)
- Metric: BLEU score on a fixed 50-sentence test set (translated reference from a human or DeepL)
- Controlled variable: same ASR output fed to both MT engines
- [ ] Prepare 50-sentence EN→HIN test set with reference translations
- [ ] Run BLEU evaluation for both engines
- [ ] Record: BLEU, translation latency per sentence (ms)

#### 3C — TTS Engine Comparison
- Engines: MMS-TTS (CUDA) vs MMS-TTS (CPU) vs AI4Bharat IndicF5 (if available)
- Metric: RTF (real-time factor), Mean Opinion Score proxy (5-point informal rating by 3 listeners)
- [ ] Record RTF for 10 fixed Hindi sentences each
- [ ] Collect MOS proxy ratings

#### 3D — End-to-End Latency Breakdown
- Measure per-stage latency: ASR → MT → TTS → audio playback start
- Run on: GPU machine + CPU-only machine (to show deployment range)
- 100-sentence continuous session, report mean ± std per stage
- [ ] Implement per-stage timing instrumentation if not already present
- [ ] Run on both hardware configs and record results

#### 3E — Baseline Comparison (REQUIRED for journal)
- Baseline A: Google Translate web + gTTS (cloud, online) — show accuracy parity at zero internet dependency
- Baseline B: Simple cascade (Vosk ASR + MarianMT + pyttsx3) — show quality improvement of current stack
- Metric: WER, BLEU, RTF, MOS proxy — same as above, same test sets
- [ ] Implement Baseline B as a standalone script
- [ ] Record all metrics for both baselines using the same test sets as 3A–3C

#### 3F — Overlap Correction Ablation
- Compare: pipeline with tail-replacement (Phase 1) vs without
- Metric: Repetition Rate (% of sentences with audible duplicate phrases, rated by human listener)
- [ ] Define Repetition Rate measurement methodology
- [ ] Run both conditions on the same 2-min audio session
- [ ] Record Repetition Rate for each condition

### Deliverables
- `results/` directory with raw CSV outputs for each experiment
- `benchmarks/` scripts that are reproducible (documented in README)

---

## Phase 4 — Adaptive Deployment Tiers
**Goal:** Make the pipeline viable on low-end CPU devices, not just CUDA workstations.
**Relevance to journal:** Strengthens the "accessible, offline, low-resource" framing.

### Tiers
| Tier | Hardware | ASR | MT | TTS |
|---|---|---|---|---|
| GPU | CUDA GPU ≥ 4GB VRAM | faster-whisper medium | Argos / MarianMT | MMS-TTS (CUDA) |
| CPU-High | Modern CPU, 8+ GB RAM | faster-whisper small.en int8 | Argos | MMS-TTS (CPU) |
| CPU-Low | Low-end CPU, 4 GB RAM | faster-whisper base.en int8 | Argos (lightweight) | AI4Bharat IndicTTS or espeak |

### Tasks
- [ ] Implement auto-detection of available hardware on startup
- [ ] Select model config automatically based on detected tier
- [ ] Validate CPU-Low tier runs in real-time (RTF < 1.0) on a constrained machine
- [ ] Document tier selection logic in README

---

## Phase 5 — Paper Writing (IEEE Access)
**Goal:** Submit to IEEE Access. Paper is a journal article, not a conference short paper.

### Paper Structure (target ~8,000 words)
1. Abstract
2. Introduction — language barrier problem, Indic language gap, offline constraint motivation
3. Related Work — prior S2ST systems, Whisper/MMS-TTS/Argos literature, Indic NLP landscape
4. System Architecture — cascade design, component interfaces, overlap correction design
5. Experimental Setup — hardware, datasets, metrics definitions
6. Results — tables from Phase 3 experiments (3A–3F), all with baselines
7. Discussion — tradeoffs, failure modes, what Phase 2 languages revealed
8. Limitations — current language scope, MOS proxy vs full MOS study, latency on very low-end hardware
9. Future Work — any-to-any language support, P2P real-time streaming, end-to-end neural S2ST
10. Conclusion

### Limitations to be honest about
- MOS proxy is informal; a full crowd-sourced MOS study is out of scope
- EN→HIN is the primary evaluated pair; other Indic pairs are preliminary
- Any-to-any language is a future goal, not a current claim

### Future Work section must include
- Any-to-any language S2ST (the long-term vision, acknowledged here but deferred)
- P2P real-time streaming architecture
- End-to-end neural S2ST (E2E models like SeamlessM4T as potential replacement for cascade)
- On-device quantised models for sub-1W edge hardware

### Submission Checklist
- [ ] IEEE LaTeX template (`IEEEtran`) — use Overleaf
- [ ] All figures at 300 DPI minimum, IEEE column width
- [ ] All tables reproducible from `results/` CSVs
- [ ] Author affiliations and ORCID IDs confirmed
- [ ] Check institution IEEE Open Access agreement (waives $1,995 APC)
- [ ] Cover letter drafted
- [ ] Verify journal scope match: IEEE Access — broad applied engineering, open access

---

## Version Tags

| Tag | Meaning |
|---|---|
| `v0.1` | Phase 0 complete — working EN→HIN pipeline |
| `v0.2` | Phase 1 complete — overlap fix landed, regression tests passing |
| `v0.3` | Phase 2 complete — minimum 2 Indic language pairs working |
| `v1.0` | Phase 3 complete — all benchmarks run, paper-ready results |
| `v1.0-submit` | Phase 5 complete — IEEE Access submission ready |

---

*PHASES.md is a living document. Update task checkboxes on each merge to develop.*