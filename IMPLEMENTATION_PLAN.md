# PolyglotTalk — Implementation Plan

> Real-time, fully offline Speech-to-Speech Translation (S2ST) desktop
> application. Mic → Transcribe → Translate → Speak. Target latency:
> 3–4 seconds end-to-end on CPU-only Windows.

---

## 1. Component Selection Rationale

### 1.1 ASR Layer

| Criterion | faster-whisper (base, int8) | Vosk (~50 MB) | Whisper.cpp |
|---|---|---|---|
| Accuracy | **High** — Whisper-class WER | Moderate — older architecture | High — same Whisper weights |
| Latency (2.5 s chunk) | ~0.8–1.2 s on 4-core CPU | ~0.3–0.5 s | ~0.6–1.0 s |
| RAM | ~350 MB (int8 model) | ~120 MB | ~300 MB |
| Python integration | Native — `faster_whisper.WhisperModel` | Native — `vosk.KaldiRecognizer` | Requires ctypes/subprocess wrapper |
| Setup on Windows | `pip install faster-whisper` + auto-downloads model | `pip install vosk` + manual model download | Build from source or find prebuilt wheel |
| Streaming support | Returns **generator** of `Segment` objects — must be consumed with `for` loop | True streaming via `AcceptWaveform()` | Batch only |

**Recommendation: `faster-whisper` (base.en, int8, cpu)**

- Best accuracy-to-latency ratio for 2.5 s chunks.
- CTranslate2 backend gives int8 quantization out of the box.
- Generator-based output requires draining all segments per chunk
  (plan must account for this — see §5).
- `beam_size=1` + `vad_filter=False` for lowest latency on short chunks.

**Fallback: `Vosk`** — if RAM is critically constrained (< 1 GB free)
or faster-whisper has DLL issues on the target machine. Vosk's
`AcceptWaveform` API is simpler but accuracy is notably lower for
accented speech.

---

### 1.2 Translation Layer

| Criterion | MarianMT (HuggingFace) | Argos Translate | ONNX MarianMT |
|---|---|---|---|
| Quality | **High** — Helsinki-NLP trained | Good — OpenNMT CTranslate2 | Same as MarianMT |
| Model size | ~300 MB per language pair | ~100 MB per language pair | ~150 MB (quantized) |
| Latency (1 sentence) | ~0.4–0.8 s (num_beams=1) | ~0.2–0.4 s (CTranslate2 backend) | ~0.15–0.3 s |
| Setup | `pip install transformers sentencepiece` + auto-download | `pip install argostranslate` + `argostranslate.package.install_from_path()` | Manual ONNX export step required |
| Offline model prep | `AutoModelForSeq2SeqLM.from_pretrained()` caches to `~/.cache/huggingface` | Download `.argosmodel` file, install via API | Export once, load `.onnx` file |
| Context prefix support | Easy — prepend to input string | Easy — prepend to input string | Easy — prepend to input string |

**Recommendation: `Argos Translate`**

- Smallest model footprint (~100 MB per pair).
- Built on CTranslate2 internally — already optimized for CPU.
- `pip install argostranslate` works cleanly on Windows.
- Simple API: `argostranslate.translate.translate(text, from, to)`.
- No need for `transformers` / `torch` heavy stack just for MT
  (faster-whisper already brings CTranslate2).
- Offline model installation is a single function call with a
  downloaded `.argosmodel` file.

**Fallback: `MarianMT` via HuggingFace** — if Argos quality is
insufficient for the language pair or more language pairs are needed.
Use `num_beams=1`, `max_length=128` for speed.

---

### 1.3 TTS Layer

| Criterion | pyttsx3 (SAPI5) | Coqui TTS | Kokoro TTS |
|---|---|---|---|
| Voice quality | Robotic — Windows built-in voices | **Neural** — near-human | Good neural — better than pyttsx3 |
| Latency | **< 50 ms** — instant | 2–5 s on CPU | ~0.5–1.5 s on CPU |
| Model size | 0 MB (system voices) | ~500 MB | ~82 MB |
| Setup | `pip install pyttsx3` | `pip install TTS` (heavy deps) | `pip install kokoro` |
| Windows compat | **Native** — uses SAPI5 COM | Requires espeak-ng install | Pure Python — works |
| Thread safety | **Must init inside owning thread** — COM threading model | Thread-safe | Thread-safe |
| Hindi/multilingual | Only if Windows language pack installed | Model-dependent | Limited language support |

**Recommendation: `pyttsx3` (SAPI5 backend)**

- Zero additional model download — uses Windows system voices.
- Sub-50 ms synthesis latency keeps total pipeline under 3 s.
- For Hindi output, requires the Windows Hindi language pack to be
  installed (Settings → Time & Language → Add a language → Hindi →
  install Speech).
- **Critical constraint**: `pyttsx3.init()` must happen inside the TTS
  thread, never in main thread. SAPI5 uses COM, and the COM apartment
  must match the thread that calls `runAndWait()`.

**Fallback: `Kokoro TTS`** — if voice quality is unacceptable. ~82 MB
model, ~1 s latency, better quality. Increases total pipeline latency
to ~4–5 s.

---

### 1.4 Audio Capture

**Choice: `sounddevice`** (no alternatives evaluated — standard choice)

- `sounddevice.InputStream` with `samplerate=16000, channels=1, dtype='float32'`.
- Callback-based capture pushes raw numpy arrays into the audio queue.
- `blocksize = 16000 * 2.5 = 40000` samples per 2.5 s chunk.

---

### 1.5 Threading

**Choice: `threading` + `queue.Queue`** (no alternatives — per constraints)

- Four daemon threads + main thread for orchestration.
- `queue.Queue(maxsize=N)` between each stage for backpressure.
- Sentinel value `None` pushed to each queue for clean shutdown.

---

## 2. File Structure

```
polyglot-talk/
├── IMPLEMENTATION_PLAN.md        # This document
├── requirements.txt              # Pinned dependencies
├── setup_models.py               # One-time model download script
├── main.py                       # Entry point — wires pipeline, handles Ctrl+C
├── config.py                     # Constants: sample rate, chunk duration, model paths, queue sizes
├── audio_capture.py              # Mic input thread
├── asr_engine.py                 # faster-whisper transcription thread
├── translator.py                 # Argos Translate translation thread (+ context continuity)
├── tts_engine.py                 # pyttsx3 speech synthesis thread
├── pipeline.py                   # Pipeline orchestrator — builds threads, queues, starts/stops
└── tests/
    ├── test_audio_capture.py     # Record 5 s, save to WAV, verify non-silence
    ├── test_asr.py               # Transcribe a known WAV file, assert expected text
    ├── test_translator.py        # Translate known sentences, assert expected output
    ├── test_tts.py               # Speak a hardcoded string, verify no crash
    ├── test_context.py           # Unit tests for context prefix/trim logic
    └── test_pipeline_e2e.py      # Full pipeline with synthetic audio, measure latency
```

| File | Responsibility |
|---|---|
| `main.py` | Parse CLI args (source lang, target lang), instantiate `Pipeline`, register signal handler, block on `KeyboardInterrupt` |
| `config.py` | All constants in one place: `SAMPLE_RATE=16000`, `CHUNK_DURATION=2.5`, `QUEUE_MAXSIZE=2`, model identifiers, log format |
| `audio_capture.py` | Open mic stream, collect 2.5 s chunks, push `np.ndarray` into `audio_queue` |
| `asr_engine.py` | Load faster-whisper model once, consume from `audio_queue`, drain generator, push transcribed `str` into `text_queue` |
| `translator.py` | Load Argos model once, maintain context deque, consume from `text_queue`, push translated `str` into `tts_queue` |
| `tts_engine.py` | Init pyttsx3 inside thread, consume from `tts_queue`, call `engine.say()` + `engine.runAndWait()` |
| `pipeline.py` | Create queues, instantiate all 4 workers, start threads, provide `stop()` method |
| `setup_models.py` | Download faster-whisper model + Argos language pack to local dirs for fully offline use |

---

## 3. Class & Interface Design

### 3.1 `config.py` — Module-level constants (no class)

```
SAMPLE_RATE: int = 16000
CHUNK_DURATION: float = 2.5
BLOCK_SIZE: int = int(SAMPLE_RATE * CHUNK_DURATION)  # 40000
QUEUE_MAXSIZE: int = 2
ASR_MODEL_SIZE: str = "base.en"
ASR_COMPUTE_TYPE: str = "int8"
ASR_BEAM_SIZE: int = 1
SOURCE_LANG: str = "en"
TARGET_LANG: str = "hi"
CONTEXT_MAXLEN: int = 2
TTS_RATE: int = 175          # words per minute for pyttsx3
LOG_FORMAT: str = "[%(asctime)s %(threadName)s] %(message)s"
```

---

### 3.2 `AudioCapture` (in `audio_capture.py`)

```
class AudioCapture:
    """Captures microphone audio in fixed-size chunks."""

    __init__(self, audio_queue: queue.Queue[np.ndarray],
             stop_event: threading.Event,
             sample_rate: int = SAMPLE_RATE,
             chunk_duration: float = CHUNK_DURATION)

    run(self) -> None
        # Thread target. Opens sounddevice.InputStream,
        # collects samples into a buffer, pushes complete
        # chunks (np.ndarray, float32, shape=(BLOCK_SIZE,))
        # into audio_queue. Exits when stop_event is set.

    _audio_callback(self, indata: np.ndarray, frames: int,
                    time_info, status) -> None
        # sounddevice callback — appends indata to internal buffer.
        # When buffer >= BLOCK_SIZE, pushes chunk to audio_queue.
```

- **Reads from**: microphone hardware (via sounddevice)
- **Writes to**: `audio_queue`

---

### 3.3 `ASREngine` (in `asr_engine.py`)

```
class ASREngine:
    """Transcribes audio chunks using faster-whisper."""

    __init__(self, audio_queue: queue.Queue[np.ndarray],
             text_queue: queue.Queue[str],
             stop_event: threading.Event,
             model_size: str = ASR_MODEL_SIZE,
             compute_type: str = ASR_COMPUTE_TYPE,
             beam_size: int = ASR_BEAM_SIZE)
        # Loads WhisperModel HERE (not in run()).
        # self.model = WhisperModel(model_size, device="cpu",
        #                           compute_type=compute_type)

    run(self) -> None
        # Thread target. Loops: get chunk from audio_queue,
        # call self._transcribe(chunk), push result to text_queue.
        # Exits when stop_event is set and audio_queue is empty.

    _transcribe(self, audio: np.ndarray) -> str
        # Calls self.model.transcribe(audio, beam_size=...,
        #   language="en", vad_filter=False)
        # Returns (segments, info) — segments is a GENERATOR.
        # Must drain generator: "".join(seg.text for seg in segments)
        # Returns concatenated text, stripped.
        # If result is empty or whitespace-only, returns "".
```

- **Reads from**: `audio_queue`
- **Writes to**: `text_queue`

---

### 3.4 `Translator` (in `translator.py`)

```
class Translator:
    """Translates text using Argos Translate with context continuity."""

    __init__(self, text_queue: queue.Queue[str],
             tts_queue: queue.Queue[str],
             stop_event: threading.Event,
             source_lang: str = SOURCE_LANG,
             target_lang: str = TARGET_LANG,
             context_maxlen: int = CONTEXT_MAXLEN)
        # Loads Argos translation model HERE.
        # self._context: collections.deque[str] = deque(maxlen=context_maxlen)
        # self._installed_lang: argostranslate.translate.Language (cached)

    run(self) -> None
        # Thread target. Loops: get text from text_queue,
        # skip if empty, call self._translate_with_context(text),
        # push result to tts_queue.

    _translate_with_context(self, text: str) -> str
        # 1. Build prefix = " ".join(self._context)
        # 2. Build input  = f"{prefix} {text}".strip() if prefix else text
        # 3. raw_output = argostranslate.translate.translate(input, src, tgt)
        # 4. Trim prefix: translate prefix alone → prefix_translated,
        #    if raw_output starts with prefix_translated, strip it.
        # 5. Append trimmed result to self._context
        # 6. Return trimmed result

    _load_model(self) -> None
        # Ensures the Argos language package for source→target is installed.
        # Raises RuntimeError if package not found.
```

- **Reads from**: `text_queue`
- **Writes to**: `tts_queue`
- **Owns**: `self._context` deque (context continuity state)

---

### 3.5 `TTSEngine` (in `tts_engine.py`)

```
class TTSEngine:
    """Speaks translated text aloud using pyttsx3 / SAPI5."""

    __init__(self, tts_queue: queue.Queue[str],
             stop_event: threading.Event,
             rate: int = TTS_RATE)
        # Does NOT init pyttsx3 here. Stores params only.
        # self._rate = rate
        # self._engine: pyttsx3.Engine | None = None

    run(self) -> None
        # Thread target.
        # FIRST LINE: self._engine = pyttsx3.init()  ← inside thread!
        # Configure: self._engine.setProperty('rate', self._rate)
        # Optionally: select voice by language ID.
        # Loop: get text from tts_queue, call self._speak(text).
        # On exit: self._engine.stop()

    _speak(self, text: str) -> None
        # self._engine.say(text)
        # self._engine.runAndWait()
        # Logs the text spoken and duration.
```

- **Reads from**: `tts_queue`
- **Writes to**: speakers (via SAPI5)

---

### 3.6 `Pipeline` (in `pipeline.py`)

```
class Pipeline:
    """Wires together all stages and manages lifecycle."""

    __init__(self, source_lang: str, target_lang: str)
        # Creates queues, instantiates workers, does NOT start threads.
        # self.audio_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
        # self.text_queue  = queue.Queue(maxsize=QUEUE_MAXSIZE)
        # self.tts_queue   = queue.Queue(maxsize=QUEUE_MAXSIZE)
        # self.stop_event  = threading.Event()
        # self.audio_capture = AudioCapture(...)
        # self.asr_engine    = ASREngine(...)
        # self.translator    = Translator(...)
        # self.tts_engine    = TTSEngine(...)

    start(self) -> None
        # Creates and starts 4 daemon threads.
        # Prints "Pipeline started. Speak now..."

    stop(self) -> None
        # Sets stop_event. Pushes sentinel None into each queue.
        # Joins all threads with timeout=5.
        # Prints "Pipeline stopped."

    wait(self) -> None
        # Blocks main thread until KeyboardInterrupt.
        # Calls self.stop() on interrupt.
```

- **Owns**: all queues, all worker instances, all threads, stop_event

---

### 3.7 `main.py` — Entry point (no class)

```
main() -> None
    # 1. Parse args: --source en --target hi
    # 2. Configure logging
    # 3. Print banner
    # 4. pipeline = Pipeline(source_lang, target_lang)
    # 5. pipeline.start()
    # 6. pipeline.wait()      ← blocks here
```

---

## 4. Threading & Queue Design

### 4.1 Queue Inventory

| Queue Name | Type | maxsize | Producer Thread | Consumer Thread | Item Type |
|---|---|---|---|---|---|
| `audio_queue` | `queue.Queue` | 2 | `AudioCaptureThread` | `ASRThread` | `np.ndarray` (float32, shape `(40000,)`) or `None` (sentinel) |
| `text_queue` | `queue.Queue` | 2 | `ASRThread` | `TranslatorThread` | `str` (transcribed text) or `None` (sentinel) |
| `tts_queue` | `queue.Queue` | 2 | `TranslatorThread` | `TTSThread` | `str` (translated text) or `None` (sentinel) |

### 4.2 Thread Inventory

| Thread Name | Target | Daemon | Reads | Writes |
|---|---|---|---|---|
| `AudioCaptureThread` | `AudioCapture.run` | Yes | Microphone | `audio_queue` |
| `ASRThread` | `ASREngine.run` | Yes | `audio_queue` | `text_queue` |
| `TranslatorThread` | `Translator.run` | Yes | `text_queue` | `tts_queue` |
| `TTSThread` | `TTSEngine.run` | Yes | `tts_queue` | Speakers |

### 4.3 Backpressure Strategy

- All queues have `maxsize=2`.
- **Producers use `queue.put(item, timeout=1.0)`** inside a loop that
  checks `stop_event`. If the put times out (queue full), re-check
  `stop_event` and retry. This means:
  - If TTS falls behind, `tts_queue` fills up → `TranslatorThread`
    blocks on put → `text_queue` fills up → `ASRThread` blocks on put
    → `audio_queue` fills up → `AudioCapture` drops the current chunk
    (logs a warning: `"Dropping audio chunk — pipeline backed up"`).
  - This is the correct behavior: we drop the oldest unprocessed audio
    rather than accumulating unbounded memory.
- **Consumers use `queue.get(timeout=0.5)`** inside a loop that checks
  `stop_event`. `queue.Empty` exceptions are silently retried.

### 4.4 Shutdown Signal Strategy

1. User presses `Ctrl+C` → `KeyboardInterrupt` caught in `Pipeline.wait()`.
2. `Pipeline.stop()` is called:
   - `stop_event.set()` — all threads see this on next loop iteration.
   - Push `None` sentinel into each queue to unblock any thread sitting
     in `queue.get()`.
   - `thread.join(timeout=5)` for each thread in reverse order:
     `TTSThread` → `TranslatorThread` → `ASRThread` → `AudioCaptureThread`.
3. Each thread's `run()` loop has this structure:
   ```
   while not self.stop_event.is_set():
       try:
           item = self.input_queue.get(timeout=0.5)
       except queue.Empty:
           continue
       if item is None:       # sentinel
           break
       # ... process item ...
   ```
4. Daemon threads ensure process exits even if join times out.

---

## 5. Context Continuity Design

### 5.1 State Location

- The context deque lives in `Translator` as `self._context: deque[str]`.
- `maxlen=2` — stores the last 2 **translated output** strings.
- Only `TranslatorThread` reads/writes `self._context` — no lock needed
  (single-writer, single-reader, same thread).

### 5.2 Prefix Injection (before translation)

```
Step 1:  prefix_source = " ".join(self._context_source)
         # _context_source: deque of last 2 SOURCE (English) segments
Step 2:  full_input = f"{prefix_source} ||| {new_text}" if prefix_source else new_text
         # The " ||| " separator helps the MT model distinguish context
         # from new input. If Argos doesn't handle this well, fall back
         # to plain concatenation: f"{prefix_source}. {new_text}"
```

**Revised approach (simpler, more reliable with Argos):**

```
Step 1:  prefix_source = " ".join(self._context_source)
Step 2:  combined_input = f"{prefix_source} {new_text}".strip()
Step 3:  full_translation = translate(combined_input, src, tgt)
```

### 5.3 Prefix Trimming (after translation)

```
Step 4:  prefix_translation = translate(prefix_source, src, tgt) if prefix_source else ""
Step 5:  if full_translation.startswith(prefix_translation):
             trimmed = full_translation[len(prefix_translation):].strip()
         else:
             # Fuzzy fallback: use difflib.SequenceMatcher to find
             # longest common prefix between full_translation and
             # prefix_translation. Trim up to that point.
             trimmed = _fuzzy_trim(full_translation, prefix_translation)
Step 6:  self._context_source.append(new_text)
         # Store source text (not translated) for next prefix
Step 7:  return trimmed if trimmed else full_translation
         # Safety: never return empty string
```

### 5.4 Edge Cases

| Edge Case | Handling |
|---|---|
| `prefix_source` longer than `new_text` | No special handling needed — deque(maxlen=2) limits prefix to at most 2 segments (~2–6 sentences). If combined input exceeds MT model's max token limit (512), truncate `prefix_source` from the left. |
| `new_text` is empty or whitespace | Skip translation entirely, do not push to `tts_queue`, do not update context. |
| `prefix_translation` not a prefix of `full_translation` | Use fuzzy trim (difflib). If overlap < 30% of `prefix_translation` length, discard prefix trimming and return `full_translation` as-is. Log a warning. |
| First 1–2 chunks (context empty) | `prefix_source` is `""`, `combined_input = new_text`, no trimming needed. |
| Repeated/hallucinated ASR output (e.g., faster-whisper repeats "Thank you" on silence) | Check if `new_text == previous_text` (store last raw ASR output). If identical, skip. |
| `prefix_translation` cache | Cache the translated prefix in `self._cached_prefix_translation` to avoid re-translating the same prefix string on consecutive calls. Invalidate when `self._context_source` changes. |

---

## 6. Startup Sequence

| Step | Action | Console Output | Est. Time |
|---|---|---|---|
| 1 | `main.py` parses CLI args | `PolyglotTalk v0.1 — Offline S2ST` | < 10 ms |
| 2 | Configure `logging` module | _(none)_ | < 10 ms |
| 3 | `Pipeline.__init__()` begins | `[init] Creating pipeline: en → hi` | < 10 ms |
| 4 | Create 3 queues + stop_event | _(none)_ | < 10 ms |
| 5 | `ASREngine.__init__()` — loads faster-whisper model | `[init] Loading ASR model (base.en, int8)...` → `[init] ASR model loaded in 3.2s` | **2–5 s** |
| 6 | `Translator.__init__()` — loads Argos model | `[init] Loading translation model (en → hi)...` → `[init] Translation model loaded in 1.5s` | **1–3 s** |
| 7 | `AudioCapture.__init__()` — no heavy init | _(none)_ | < 10 ms |
| 8 | `TTSEngine.__init__()` — no heavy init (pyttsx3 inits in thread) | _(none)_ | < 10 ms |
| 9 | `Pipeline.start()` — launches 4 threads | `[init] Starting AudioCaptureThread...` | < 10 ms |
| 10 | `TTSThread` starts → `pyttsx3.init()` inside thread | `[init] TTS engine initialized (SAPI5)` | ~0.5 s |
| 11 | `AudioCaptureThread` opens mic stream | `[init] Microphone stream opened (16000 Hz, mono)` | ~0.2 s |
| 12 | Pipeline ready | **`✓ Pipeline ready. Speak now... (Ctrl+C to stop)`** | — |

**Total startup time: ~5–10 seconds** (dominated by model loading).

**Model loading order rationale**: ASR and Translator are loaded
sequentially in `Pipeline.__init__()` (main thread) before any threads
start. This ensures models are fully in memory before the first audio
chunk arrives. TTS init happens in its own thread because of SAPI5 COM
requirements, but pyttsx3.init() is fast (~0.5 s).

---

## 7. Known Pitfalls & Mitigations

| # | Pitfall | Mitigation |
|---|---|---|
| 1 | **pyttsx3 COM threading violation** — initializing pyttsx3 in main thread then calling `runAndWait()` in another thread causes `COMError` or silent hangs. | Always call `pyttsx3.init()` as the first line inside `TTSEngine.run()`, never in `__init__()` or main thread. |
| 2 | **faster-whisper generator not drained** — `model.transcribe()` returns `(generator, info)`. If the generator is not fully consumed (e.g., only taking the first segment), CTranslate2 may leak memory or produce corrupt state on the next call. | Always drain the generator: `text = " ".join(seg.text for seg in segments)`. Never break out of the segment loop early. |
| 3 | **sounddevice PortAudio DLL not found on Windows** — `sounddevice` ships its own PortAudio DLL, but some Windows installs or virtualenvs fail to locate it. | Pin `sounddevice>=0.4.6` (bundles `_sounddevice_data`). If import fails, fall back to `pip install sounddevice --force-reinstall`. Test import in `setup_models.py`. |
| 4 | **Argos Translate model not pre-installed** — calling `argostranslate.translate.translate()` without first installing the language package raises a lookup error with a cryptic message. | `setup_models.py` must download and install the `.argosmodel` file. At startup, `Translator.__init__()` verifies the package is installed and raises `RuntimeError("Run setup_models.py first")` if missing. |
| 5 | **faster-whisper hallucinations on silence** — Whisper models tend to hallucinate repeated phrases ("Thank you", "you", "Bye") when receiving silent or near-silent audio. | Before pushing to `text_queue`, check: (a) audio RMS energy > threshold (e.g., `np.sqrt(np.mean(audio**2)) > 0.01`), and (b) transcribed text is not identical to the previous transcription. Skip if either check fails. |
| 6 | **Windows default mic not being 16 kHz** — most Windows mics default to 44100 or 48000 Hz. `sounddevice.InputStream(samplerate=16000)` forces resampling in PortAudio, but some drivers reject non-native rates. | Let sounddevice open at the device's native rate, then resample to 16 kHz in the callback using `scipy.signal.resample` or `librosa.resample`. Alternatively, query `sd.query_devices()` at startup and log the native rate. |
| 7 | **Queue deadlock on shutdown** — if a thread is blocked on `queue.put()` (full queue) when `stop_event` is set, it never checks the event and the join times out. | Use `queue.put(item, timeout=1.0)` in a loop, checking `stop_event` between retries. During shutdown, additionally drain all queues after setting `stop_event` to unblock any stuck puts. |
| 8 | **CTranslate2 thread contention** — both faster-whisper (ASR) and Argos Translate use CTranslate2 internally. If both call CT2 simultaneously (on pipeline startup overlap), thread-pool sizing can conflict. | Set `OMP_NUM_THREADS=2` and `CT2_INTER_THREADS=1` environment variables before importing either library. This limits each CT2 instance to 2 threads, avoiding over-subscription on a 4-core CPU. Set these in `config.py` at import time via `os.environ`. |

---

## 8. Testing Strategy

### 8.1 Component-Level Tests

| Test File | What It Verifies | How to Run |
|---|---|---|
| `test_audio_capture.py` | Records 3 seconds from mic, saves to `test_output.wav`, asserts file size > 0 and RMS > 0.001 (not silence). | `python -m tests.test_audio_capture` — requires live mic. |
| `test_asr.py` | Loads faster-whisper model, transcribes a bundled `test_audio/hello.wav` (2.5 s clip of "Hello, how are you?"), asserts `"hello"` appears in output (case-insensitive). | `python -m tests.test_asr` — no mic needed. |
| `test_translator.py` | Translates `"Hello, how are you?"` from en→hi, asserts output is non-empty, is not ASCII-only (contains Devanagari), and length > 5 chars. | `python -m tests.test_translator` |
| `test_tts.py` | Inits pyttsx3 inside a child thread, speaks `"Testing one two three"`, asserts no exception raised. Runs in < 5 s. | `python -m tests.test_tts` — requires speakers/headphones. |
| `test_context.py` | Unit tests for `Translator._translate_with_context()` using mocked `translate()` calls. Tests: empty context, 1-segment context, 2-segment context, prefix trimming, fuzzy trimming, empty input skip, repeated input skip. | `python -m pytest tests/test_context.py` |

### 8.2 End-to-End Test

**File**: `test_pipeline_e2e.py`

1. Create a `Pipeline` with a modified `AudioCapture` that reads from a
   WAV file instead of the mic (dependency injection via a `source`
   parameter or subclass).
2. Replace `TTSEngine._speak()` with a mock that appends text to a list.
3. Feed 3 consecutive 2.5 s chunks of known English speech.
4. Assert:
   - At least 2 translated strings appear in the mock's list within 15 s.
   - Each translated string is non-empty.
   - No thread raised an unhandled exception.
5. Call `pipeline.stop()`, assert all threads terminated within 5 s.

**A successful E2E test** means: WAV audio in → non-empty Hindi text
out via mock TTS, no crashes, all threads exit cleanly.

### 8.3 Latency Measurement

Each worker class logs timestamps at stage entry and exit:

```
[2026-02-22 10:00:01.234 ASRThread] Chunk received
[2026-02-22 10:00:02.456 ASRThread] Transcription done (1.222s): "Hello how are you"
[2026-02-22 10:00:02.460 TranslatorThread] Text received
[2026-02-22 10:00:02.870 TranslatorThread] Translation done (0.410s): "नमस्ते आप कैसे हैं"
[2026-02-22 10:00:02.875 TTSThread] Text received
[2026-02-22 10:00:02.920 TTSThread] Speech done (0.045s)
```

- Each `_transcribe`, `_translate_with_context`, and `_speak` method
  records `t0 = time.perf_counter()` at entry and
  `elapsed = time.perf_counter() - t0` at exit.
- `AudioCapture` timestamps when each chunk is pushed; `TTSEngine`
  timestamps when speech finishes. The difference is the end-to-end
  latency, logged as: `[pipeline] E2E latency: 3.12s`.
- To correlate chunks across threads, attach a monotonically increasing
  `chunk_id: int` to each queue item (use a `NamedTuple` or
  `dataclass` wrapper instead of raw `str`/`ndarray`).

**Chunk wrapper dataclasses:**

```
@dataclass
class AudioChunk:
    chunk_id: int
    audio: np.ndarray
    timestamp: float          # time.perf_counter() when captured

@dataclass
class TextSegment:
    chunk_id: int
    text: str
    timestamp: float          # time.perf_counter() when ASR finished

@dataclass
class TranslatedSegment:
    chunk_id: int
    text: str
    timestamp: float          # time.perf_counter() when translation finished
```

These replace the raw types in the queues:
- `audio_queue: Queue[AudioChunk | None]`
- `text_queue: Queue[TextSegment | None]`
- `tts_queue: Queue[TranslatedSegment | None]`

---

## 9. Future Upgrade Path

### 9.1 Swap pyttsx3 → Kokoro TTS

1. Create `tts_kokoro.py` with a `KokoroTTSEngine` class that has the
   same interface: `__init__(tts_queue, stop_event, ...)` and `run()`.
2. Inside `run()`, load the Kokoro model (instead of `pyttsx3.init()`).
3. `_speak(text)` generates a WAV numpy array via Kokoro, then plays it
   via `sounddevice.play(wav, samplerate=...)`.
4. In `pipeline.py`, change the import:
   `from tts_kokoro import KokoroTTSEngine as TTSEngine`.
5. No changes needed in any other module — same queue contract.
6. Update `config.py` with `KOKORO_MODEL_PATH` and `KOKORO_SAMPLE_RATE`.
7. Update `setup_models.py` to download the Kokoro model (~82 MB).

### 9.2 Swap Argos Translate → ONNX MarianMT

1. **One-time export step**: script `export_marian_onnx.py` that loads
   `Helsinki-NLP/opus-mt-en-hi` via transformers, exports encoder +
   decoder to ONNX using `torch.onnx.export()`, then quantizes with
   `onnxruntime.quantization`.
2. Create `translator_onnx.py` with `ONNXTranslator` class, same
   interface as `Translator`.
3. `__init__()` loads `onnxruntime.InferenceSession(encoder.onnx)` and
   `InferenceSession(decoder.onnx)`, plus `AutoTokenizer`.
4. `_translate_with_context()` tokenizes → runs encoder session →
   runs decoder session with greedy decoding → detokenizes.
5. Context continuity logic (`_context`, prefix, trimming) is
   **identical** — copy from `translator.py`.
6. In `pipeline.py`, swap import. No other module changes.

### 9.3 Add Voice Activity Detection (VAD) Pre-filter

1. Add `pip install silero-vad` or `pip install webrtcvad` to
   requirements.
2. Create `vad_filter.py` with class `VADFilter`:
   ```
   __init__(self, audio_queue_in, audio_queue_out, stop_event)
   run(self) → None
       # Consumes from audio_queue_in, runs VAD,
       # pushes to audio_queue_out only if speech detected.
   ```
3. Insert between `AudioCapture` and `ASREngine` in the pipeline:
   ```
   AudioCapture → audio_queue_raw → VADFilter → audio_queue → ASREngine
   ```
4. In `pipeline.py`, add one more queue (`audio_queue_raw`) and one
   more thread (`VADThread`).
5. Benefit: eliminates silence chunks from reaching ASR, reducing
   CPU usage and preventing Whisper hallucinations.

---

## 10. Setup Instructions

### 10.1 Python Environment

```bash
# Requires uv package manager and Python 3.11.9
# Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11.9
source .venv/bin/activate
```

### 10.2 Install Dependencies

```bash
uv pip install --upgrade pip setuptools wheel

# Core pipeline
uv pip install faster-whisper==1.1.0      # ASR (includes CTranslate2)
uv pip install argostranslate==1.9.6      # Translation
uv pip install pyttsx3==2.98              # TTS (espeak-ng on Linux)
uv pip install sounddevice==0.5.1         # Audio capture
uv pip install 'numpy>=1.24,<2.0'         # Audio arrays

# Testing
uv pip install pytest==8.3.4
uv pip install 'scipy>=1.11'              # Resampling (if needed)
```

### 10.3 System-Level Dependencies

```bash
# Audio I/O (required by sounddevice)
sudo apt install -y libportaudio2 portaudio19-dev

# TTS engine (required by pyttsx3 on Linux)
sudo apt install -y espeak-ng

# Optional: mbrola Hindi voice for slightly better quality
# sudo apt install -y mbrola mbrola-hi1

# Python build tools (if any pip packages need to compile)
sudo apt install -y python3-dev build-essential
```

| Dependency | Why |
|---|---|
| **libportaudio2** | Required by sounddevice (PortAudio C library). |
| **espeak-ng** | Required by pyttsx3 as the Linux TTS backend. |
| **Working microphone** | Audio input — verify with `arecord -l`. |

> **WSL2 audio note**: Audio requires WSL2 (not WSL1) with `wslg` enabled
> (Windows 11 / Windows 10 21H2+). PulseAudio is auto-bridged. Verify
> with `pactl info` inside WSL2. If the microphone doesn't appear, run
> `wsl --update` from PowerShell and restart WSL.

No GPU drivers, CUDA, or cuDNN required.

### 10.4 Pre-download Models for Offline Use

Create and run `setup_models.py`:

```powershell
python setup_models.py
```

**What `setup_models.py` does:**

1. **faster-whisper model** — Downloads `base.en` (int8) to local
   cache. Uses `faster_whisper.WhisperModel("base.en", device="cpu",
   compute_type="int8")` which auto-downloads to
   `~/.cache/huggingface/hub/`. To force a custom path, set
   `download_root` parameter or environment variable
   `WHISPER_MODELS_DIR`.

2. **Argos Translate language pack** — Downloads the `en→hi`
   `.argosmodel` package file.
   ```python
   import argostranslate.package
   argostranslate.package.update_package_index()
   available = argostranslate.package.get_available_packages()
   pkg = next(p for p in available
              if p.from_code == "en" and p.to_code == "hi")
   pkg.install()
   ```
   After this, the model is cached in
   `~/.local/share/argos-translate/packages/` (or Windows equivalent).

3. **Verification** — After download, the script:
   - Loads the faster-whisper model and transcribes 1 second of silence
     → asserts no crash.
   - Translates `"Hello"` via Argos → asserts non-empty output.
   - Inits pyttsx3, lists available voices → prints voice names.
   - Prints `"✓ All models ready for offline use."`.

**Estimated total download**: ~450 MB (350 MB faster-whisper + 100 MB
Argos).

### 10.5 `requirements.txt`

```
faster-whisper==1.1.0
argostranslate==1.9.6
pyttsx3==2.98
sounddevice==0.5.1
numpy>=1.24,<2.0
scipy>=1.11
pytest==8.3.4
```

---

## Appendix: Data Flow Diagram

```
┌──────────────┐   audio_queue   ┌──────────────┐   text_queue   ┌──────────────┐   tts_queue   ┌──────────────┐
│              │   (maxsize=2)   │              │   (maxsize=2)  │              │   (maxsize=2) │              │
│  AudioCapture│──────────────►  │  ASREngine   │──────────────► │  Translator  │─────────────► │  TTSEngine   │
│              │  AudioChunk     │              │  TextSegment   │              │ TranslatedSeg │              │
│  (mic input) │                 │(faster-whsp) │                │(argos xlate) │               │  (pyttsx3)   │
└──────────────┘                 └──────────────┘                └──────────────┘               └──────────────┘
       ▲                                                               │
       │                                                               │
   Microphone                                              context deque(maxlen=2)
                                                          stores last 2 source segments
```

---

## Appendix: Configuration Quick Reference

| Parameter | Value | Defined In |
|---|---|---|
| `SAMPLE_RATE` | `16000` Hz | `config.py` |
| `CHUNK_DURATION` | `2.5` seconds | `config.py` |
| `BLOCK_SIZE` | `40000` samples | `config.py` |
| `QUEUE_MAXSIZE` | `2` | `config.py` |
| `ASR_MODEL_SIZE` | `"base.en"` | `config.py` |
| `ASR_COMPUTE_TYPE` | `"int8"` | `config.py` |
| `ASR_BEAM_SIZE` | `1` | `config.py` |
| `CONTEXT_MAXLEN` | `2` segments | `config.py` |
| `TTS_RATE` | `175` wpm | `config.py` |
| `OMP_NUM_THREADS` | `"2"` | `config.py` (env var) |
| `CT2_INTER_THREADS` | `"1"` | `config.py` (env var) |
