# PolyglotTalk Development Roadmap

Project: Offline Speech-to-Speech Translation (S2ST) for Indic languages  
Primary vision: A phone-to-phone translation call app — fully offline, no cloud APIs.  
You speak in your language; the other person hears you in theirs, in real time.

The desktop pipeline is a prototyping ground, not the product.  
The desktop WAV-file output model exists only because desktop mic + speaker on the same machine creates an audio feedback loop. On phones with earpieces, this doesn't exist.

---

## Phase 0 – Foundation (COMPLETE)

**Goal:** Prove the cascade works: mic → ASR → MT → TTS → audio. Offline. No cloud.

**Delivered:**

- Chunked audio capture with configurable stride/overlap (`AudioCaptureThread`)
- faster-whisper ASR (`base.en`, int8) with sentence buffering
- Argos Translate MT for EN→HIN, fully offline
- MMS-TTS (`facebook/mms-tts-hin`) on CUDA
- Per-sentence WAV output (`output/chunk_NNNN.wav`) — workaround for desktop feedback issue
- WebSocket dashboard + React frontend
- Basic overlap deduplication (prefix/suffix word matching)
- Pipeline start/stop lifecycle management

**Known debt:**

- Sentence buffer is append-only → semantic restatements cause double-speech (Phase 1 fix)
- Near-duplicate guard on full raw chunk text, not deduped text
- No word-level timestamps used despite faster-whisper supporting them
- Dashboard shows EN and HI in disconnected panes, no per-sentence pairing

**Status: DONE.**

---

## Phase 1 – ASR Quality: Fix Overlap Restatements (CURRENT)

**Goal:** Eliminate the recurring double-speech artifact from overlapping chunk boundaries.

### 1.1 Tail-replacement in sentence buffer

Problem: Overlapping chunks + append-only `_sentence_buf` produce:  
`"… and it is all and it is also converting it …"`

**Design:**

- Add `word_overlap_ratio(tail, new)` helper (Jaccard on word sets)
- Before appending new ASR text, compare against last N words of `_sentence_buf`:
  - If overlap ≥ 0.6 and `len(new) ≥ 0.7 * len(tail)` → **replace tail** (later take wins)
  - Else → append as normal
- Log replacements at DEBUG level

**Exit criteria:**

- 2+ minutes of continuous speech: zero audible semantic restatements in TTS output
- Unit tests: "and it is all" + "and it is also converting it" → only one phrasing in `SENT`

### 1.2 No mid-word flushes

- Introduce `pending_flush` state: wait one stride window before committing silence
- Strip trailing hallucination tokens (`-`, `...`) before committing a sentence

### 1.3 Paired transcript UI

- Replace split panes with a single `TranscriptFeed`: one card per sentence
- Each card: EN source row + HI translation row (skeleton while pending) + latency badge
- Move AudioSidebar + EventLog into collapsible right drawer

**Bar for completion:** This is v0.2. Ship it when double-speech is gone and the UI is readable.

---

## Phase 2 – Streaming Playback & Latency Instrumentation (Desktop)

**Goal:** Make the desktop pipeline feel live, and measure where time goes.  
This is foundational work before mobile; you need working real-time playback before P2P.

### 2.1 Streaming TTS playback

Current: TTS writes WAV files, manual playback. This is only acceptable in the prototype.

**Design:**

- `AudioPlaybackThread`: consumes TTS outputs from a queue, plays via `sounddevice`
- Maintains sentence ordering, cancels gracefully on stop
- Keep optional WAV saving for debug

**Note on the feedback problem:**  
On desktop with a single mic+speaker, enabling auto-playback will cause the speaker output to be picked up by the mic → feedback loop → ASR picks up its own TTS output.  
Solutions (pick one for testing):
  - Use headphones during development testing
  - Add a software VAD gate: suppress microphone capture while TTS is playing
  - Add a simple "TTS playing" flag: pause `AudioCaptureThread` during playback
  
The right permanent solution is **mobile with earpiece**, where this problem doesn't exist.

### 2.2 Per-stage latency instrumentation

- Timestamps for each stage: `t_capture → t_asr → t_mt → t_tts_start → t_tts_done`
- WebSocket event per sentence with these numbers
- Latency badge in transcript card: `ASR 0.8s | MT 0.1s | TTS 0.6s | Total 2.0s`
- Log as JSONL in `results/latency_log.jsonl`

**Exit criteria:**

- Latency numbers visible in UI for every sentence
- Auto-playback works with headphones connected

---

## Phase 3 – Timestamp-based Overlap Resolution (Principled Fix)

**Goal:** Replace Phase 1's Jaccard heuristic with time-axis deduplication.

**Design:**

- Enable `word_timestamps=True` in faster-whisper
- Maintain `cutoff_time` = end time of last committed word
- For each new chunk: keep only words where `mid_time > cutoff_time + ε`
- Update `cutoff_time` as words are committed

**Exit criteria:**

- Overlap duplicates eliminated without relying on text similarity
- Monotonically increasing word timestamps in logs
- No regressions on Phase 1 acceptance criteria

---

## Phase 4 – Hardware Tiers (CPU / GPU / Low-end)

**Goal:** Prove the pipeline degrades gracefully on weaker hardware —  
specifically what a student or rural user would actually run this on.

### The three tiers

| Tier | Target device | ASR model | TTS engine |
|---|---|---|---|
| **GPU** | Dev machine, CUDA ≥4 GB | `medium.en` int8 | MMS-TTS CUDA |
| **CPU-High** | Modern laptop ~8 GB RAM | `small.en` int8 | MMS-TTS CPU |
| **CPU-Low** | Budget laptop 4 GB RAM | `base.en` int8 | AI4Bharat Indic-TTS or espeak-ng |

### On approximating low-end hardware

You can use `--memory=4g` Docker or cgroups CPU pinning to stress-test robustness (crash / memory blow-up), but this does **not** accurately replicate slow single-core CPU speed of a 2019 budget phone or laptop.  
For any claim about CPU-Low tier, test on an actual low-end device eventually.

**Tasks:**

- `--tier [gpu|cpu-high|cpu-low|auto]` CLI flag
- `auto`: detect CUDA, run a short TTS RTF probe, select tier, log decision
- Run a fixed 50-sentence test per tier, capture RTF and E2E latency
- Output `results/tier_summary.csv`

**Exit criteria:**

- All three tiers run end-to-end
- CPU-Low stays below RTF ≈ 1.0 for short sentences

---

## Phase 5 – Multi-Language Support (Indic-first)

**Goal:** Architecture supports runtime language pair selection, not hard-coded EN→HI.

**Starting pairs:** EN→BEN (Bengali), EN→TAM (Tamil)

**Tasks:**

- `MMSTTS_MODEL_MAP`: ISO 639-3 → MMS-TTS model IDs
- `ARGOS_LANG_MAP`: ISO 639-3 → Argos 2-letter codes
- Translator/TTS accept `(src_lang, tgt_lang)` at runtime
- Dashboard language selector → pipeline restart with new models

**Exit criteria:**

- EN→BEN and EN→TAM run end-to-end
- Language switch from UI works cleanly (full pipeline teardown + restart)

---

## Phase 6 – P2P Call Mode (THE Core Feature)

This is what PolyglotTalk actually is: a translation call app.

### The problem being solved

Today: A speaks Hindi → gets translated to English → Person B hears English (and vice versa).  
Current barriers: speech feedback loop, single-machine pipeline, no audio transport.  
All of those go away with a phone-to-phone call model.

### 6.1 Architecture: two-phone call model

```
Phone A (Hindi speaker)                    Phone B (English speaker)
─────────────────────────                  ─────────────────────────
mic → ASR (HI) → MT (HI→EN)               mic → ASR (EN) → MT (EN→HI)
    → compressed stream ─────────────────────→ TTS (HI) → earpiece
earpiece ←── TTS (EN) ←──────────────────── compressed stream ←
```

Each phone runs the **full pipeline** for its speaker's language.  
Audio output goes to earpiece/headphone → no mic feedback.  
Network: start with LAN/WiFi; eventually mobile data.

### 6.2 Two transport modes (build in order)

**Mode A – Text transport (MT on sender):**

- Sender: ASR + MT → sends translated text
- Receiver: TTS locally
- Pros: very low bandwidth (~50–200 bytes/sentence), receiver can customise TTS voice
- Cons: receiver must have TTS for the target language installed

**Mode B – Audio transport (full pipeline on sender):**

- Sender: ASR + MT + TTS → sends Opus audio stream
- Receiver: jitter buffer + playback only
- Pros: receiver is truly thin (just a speaker), no model on receiver needed
- Cons: higher bandwidth (~10–30 kbps), more latency-sensitive

**Build Mode A first** — it's simpler and more flexible.

### 6.3 Network stack (LAN-first)

- No STUN/TURN/ICE for v1; LAN only
- Simple TCP or UDP with sequence numbers and basic jitter buffer
- Session protocol (minimal):
  - `HELLO` (src_lang, tgt_lang, mode)
  - `TEXT_FRAME` / `AUDIO_FRAME` (seq, payload)
  - `BYE`
- Upgrade to WebRTC or libp2p in a later version for NAT traversal

### 6.4 Process structure

Split the current monolith into two runnable modes:

```
python main.py --mode sender --lang hin --target en --peer 192.168.1.x
python main.py --mode receiver --lang en --peer 192.168.1.x
```

- **Sender mode:** `AudioCapture → ASR → MT → (TTS if Mode B) → P2POutThread`
- **Receiver mode:** `P2PInThread → (TTS if Mode A) → PlaybackThread`
- Logs clearly state `Mode = sender | receiver`

### 6.5 P2P metrics to instrument

- Network one-way latency (sender TTS done → receiver playback start)
- Effective E2E: speaker mic → listener earpiece
- Jitter buffer depth over time

### 6.6 Mobile target

The P2P mode should eventually run as a mobile app.  
The clearest path: **React Native + native module bridging** for audio capture/playback + Python backend via a bundled runtime, or rewrite the core pipeline in **Kotlin/Swift** using equivalent models (whisper.cpp for ASR, MMS-TTS ONNX for TTS).  
For now, build and validate P2P on desktop (two processes on the same LAN). Mobile port is Phase 7.

**Exit criteria for Phase 6:**

- Two machines on the same WiFi:
  - Machine A speaks Hindi → Machine B hears English within ≤ 3 s additional latency on top of local E2E
  - Bidirectional (both machines are simultaneously sender and receiver)
- Documented in README with exact run commands

---

## Phase 7 – Mobile App (End-game)

**Goal:** PolyglotTalk as an actual phone app, not a desktop script.

### 7.1 Architecture decision: two paths

**Path A – Python backend + thin native UI**  
- Keep Python pipeline, expose via local HTTP/gRPC
- React Native or Flutter UI shell, calls local backend
- Pros: reuse existing code; Cons: Python packaging on mobile is painful, large binary

**Path B – Native reimplementation of hot path**  
- whisper.cpp (C++) for ASR via JNI/FFI
- Argos Translate compiled for mobile (has Android support via Python-for-Android or via native port)
- MMS-TTS exported to ONNX, run via ONNX Runtime Mobile
- Pros: much smaller, proper mobile perf; Cons: more work

**Recommendation:** Start with Path A as a proof-of-concept on Android (easier to sideload), then evaluate Path B once the UX is validated.

### 7.2 The call UX

The app should feel like a regular phone call, not a "translation tool":

- Dial / receive screen
- Active call view: waveform for each speaker, running paired transcript
- Auto-detects source language or lets user set it before the call
- Works over WiFi (LAN P2P) initially; mobile data later

### 7.3 Exit criteria

- Two Android phones on WiFi: natural bilingual conversation with no more than 3–4 s perceived lag
- App installable via APK (no Play Store required initially)

---

## Phase 8 – Paper / Public Release

Write-up and release come after Phase 6 (P2P working). At that point you have something worth writing about that hasn't been done before.

**High level:**

- Tag `v1.0` once P2P on LAN is stable
- Benchmarks: E2E latency (local + P2P), WER, BLEU, RTF per tier
- Clean public repo with README, example configs, reproduce-metrics script

---

## Version milestones

| Tag | Phase complete | State |
|---|---|---|
| `v0.1` | Phase 0 | ✅ |
| `v0.2` | Phase 1 | double-speech fixed, paired UI |
| `v0.3` | Phase 2 | streaming playback, latency numbers |
| `v0.4` | Phase 3 | timestamp-based dedup |
| `v0.5` | Phase 4 | three tiers wired and measured |
| `v0.6` | Phase 5 | multi-language, runtime switching |
| `v0.7` | Phase 6 | P2P call on LAN, bidirectional |
| `v0.8` | Phase 7 | Android app, WiFi call |
| `v1.0` | Phase 8 | paper-ready, public release |