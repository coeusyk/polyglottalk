# PolyglotTalk

Real-time, fully **offline** Speech-to-Speech Translation (S2ST) running entirely on CPU — no cloud APIs, no GPU required.

```
Microphone → [Whisper ASR] → [Argos Translate] → [pyttsx3 TTS] → Speakers
```

Speak English, hear Hindi (or any supported language pair). Each stage runs in its own thread so recording, transcribing, translating, and speaking all happen **simultaneously**.

---

## Features

- **Fully offline** — all models run locally after a one-time download
- **CPU-only** — no GPU or CUDA required; works on any modern laptop
- **True pipeline parallelism** — 4 threads run concurrently with ~4 s end-to-end latency
- **Auto-stop** — pipeline exits cleanly after 2.5 s of silence
- **Live console output** — transcription and translation printed as they happen
- **Configurable** — source/target language and TTS speed via CLI flags
- **Hallucination filtering** — common Whisper silence-artifacts ("Thank you", "Bye") are blocked

---

## Demo Output

```
============================================================
 PolyglotTalk v0.1 — Offline Speech-to-Speech Translation
 EN → HI  |  CPU-only  |  No cloud APIs
============================================================
✓ Pipeline ready. Speak now… (Ctrl+C to stop)
[ASR      #1] Hello, how are you today?
[→HI      #1] नमस्ते, आप आज कैसे हैं?
[ASR      #2] I'm doing well, thank you.
[→HI      #2] मैं ठीक हूँ, धन्यवाद।
Pipeline stopped.
```

---

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.11.x |
| uv _(recommended)_ | latest |

### System packages (Linux / WSL2)

```bash
sudo apt install -y libportaudio2 portaudio19-dev espeak-ng pulseaudio libpulse0
```

| Package | Purpose |
|---|---|
| `libportaudio2` | PortAudio C library — required by `sounddevice` |
| `espeak-ng` | TTS backend for `pyttsx3` on Linux |
| `pulseaudio` / `libpulse0` | Audio routing (WSL2: connects to WSLg's RDP microphone) |

> **WSL2 users:** Audio is bridged via WSLg on Windows 11. After installing
> `libpulse0`, verify your mic appears with:
> ```bash
> PULSE_SERVER=unix:/mnt/wslg/PulseServer pactl list sources short
> ```
> You should see `RDPSource` listed. If not, run `wsl --update` from PowerShell.

> **Hindi TTS voice:** On Linux, `espeak-ng` will speak Hindi using its
> built-in voice. On Windows, install the Hindi language pack via
> *Settings → Time & Language → Language → Add Hindi → Speech*.

---

## Quick Start

### 1. Create environment and install Python dependencies

```bash
# Using uv (recommended)
uv venv --python 3.11.9
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Download models (one-time, requires internet, ~350 MB)

```bash
python setup_models.py
```

This downloads:
- `faster-whisper` `base.en` int8 model (~150 MB) → `~/.cache/huggingface/hub/`
- Argos Translate `en→hi` language pack (~100 MB) → `~/.local/share/argos-translate/`

### 3. Run

```bash
python main.py
```

Speak into your microphone. The pipeline stops automatically after ~2.5 s of silence. Press **Ctrl+C** to stop at any time.

---

## Usage

```
python main.py [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--source LANG` | `en` | Source language code |
| `--target LANG` | `hi` | Target language code |
| `--tts-rate WPM` | `175` | TTS speech speed in words per minute |
| `--log-level LEVEL` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Examples

```bash
# English → Spanish
python main.py --source en --target es

# Slower speech output
python main.py --tts-rate 140

# Debug mode (shows all internal logs)
python main.py --log-level DEBUG
```

> Language codes follow [ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).
> Available language pairs depend on which Argos Translate packages are installed.
> Run `python setup_models.py` after changing the target language.

---

## Project Structure

```
polyglot-talk/
├── main.py              # Entry point — parses CLI args, starts pipeline
├── config.py            # All constants: sample rate, model IDs, queue sizes
├── pipeline.py          # Orchestrator — wires threads and queues, manages lifecycle
├── audio_capture.py     # Thread 1: mic → AudioChunk queue
├── asr_engine.py        # Thread 2: AudioChunk → TextSegment (faster-whisper)
├── translator.py        # Thread 3: TextSegment → TranslatedSegment (Argos Translate)
├── tts_engine.py        # Thread 4: TranslatedSegment → speakers (pyttsx3)
├── models.py            # Dataclasses: AudioChunk, TextSegment, TranslatedSegment
├── setup_models.py      # One-time model download + smoke-test script
├── requirements.txt     # Pinned Python dependencies
├── IMPLEMENTATION_PLAN.md
└── tests/
    ├── test_audio_capture.py   # Live mic recording test
    ├── test_asr.py             # Whisper transcription test
    ├── test_translator.py      # Argos translation test
    ├── test_tts.py             # pyttsx3 thread-safety test
    ├── test_context.py         # Context-continuity unit tests (mocked)
    └── test_pipeline_e2e.py    # Full pipeline integration test
```

---

## How It Works

```
┌──────────────┐  audio_queue  ┌──────────────┐  text_queue  ┌──────────────┐  tts_queue  ┌──────────────┐
│ AudioCapture │  (maxsize=2)  │  ASREngine   │  (maxsize=2) │  Translator  │ (maxsize=2) │  TTSEngine   │
│  Thread 1    │ ───────────►  │  Thread 2    │ ───────────► │  Thread 3    │ ──────────► │  Thread 4    │
│  sounddevice │  AudioChunk   │faster-whisper│  TextSegment │argos-translate  TranslatedSeg│  pyttsx3     │
└──────────────┘               └──────────────┘              └──────────────┘              └──────────────┘
```

All four threads run simultaneously. At steady state:

- **Thread 1** records the next 2.5-second buffer while Thread 2 is still transcribing the current one
- **Thread 2** transcribes while Thread 3 translates the previous chunk
- **Thread 3** translates while Thread 4 is speaking the chunk before that

End-to-end latency ≈ `chunk_duration + ASR_time + MT_time` ≈ **4 seconds** — not the sum of all stage durations.

**Backpressure:** All queues use a drop-oldest strategy — if a downstream stage falls behind, the oldest unprocessed item is evicted to make room for the freshest one, so the pipeline always stays current.

**Context continuity:** The translator maintains a rolling window of the last 2 source segments, prepending them as context to each new translation to reduce sentence-boundary errors.

---

## Configuration

Key values in [config.py](config.py):

| Parameter | Default | Description |
|---|---|---|
| `SAMPLE_RATE` | `16000` Hz | Whisper requires 16 kHz mono |
| `CHUNK_DURATION` | `2.5` s | Audio buffer size per ASR call |
| `RMS_SILENCE_THRESHOLD` | `0.0001` | Below this RMS, audio is treated as silence |
| `ASR_MODEL_SIZE` | `"base.en"` | Whisper model variant |
| `ASR_COMPUTE_TYPE` | `"int8"` | Quantization (int8 = fastest on CPU) |
| `ASR_BEAM_SIZE` | `1` | Beam width (1 = greedy, fastest) |
| `QUEUE_MAXSIZE` | `2` | Max items per inter-thread queue |
| `CONTEXT_MAXLEN` | `2` | Number of past segments used as MT context |
| `TTS_RATE` | `175` | Speech rate in words per minute |

---

## Running Tests

```bash
# Run all tests
.venv/bin/python -m pytest -v

# Run with skip summary
.venv/bin/python -m pytest -v -rs

# Run a specific test file
.venv/bin/python -m pytest tests/test_translator.py -v
```

| Test | Requires | Notes |
|---|---|---|
| `test_audio_capture.py` | Live microphone | Skipped in CI or if no audio device |
| `test_asr.py` | Models installed | `hello.wav` test skipped if file absent |
| `test_translator.py` | Models installed | Verifies Devanagari output |
| `test_tts.py` | Speakers | Audibly speaks a phrase |
| `test_context.py` | Nothing | Fully mocked — runs anywhere |
| `test_pipeline_e2e.py` | Models installed | Uses synthetic audio, no mic needed |

---

## Dependencies

```
faster-whisper==1.1.0   # ASR — CTranslate2-optimised Whisper
argostranslate==1.9.6   # Machine translation — OpenNMT + CTranslate2
pyttsx3==2.98           # TTS — SAPI5 (Windows) / espeak-ng (Linux)
sounddevice==0.5.1      # Microphone input via PortAudio
numpy>=1.24,<2.0        # Audio arrays
scipy>=1.11             # Signal processing utilities
pytest==8.3.4           # Testing
```

---

## Future Upgrades

- **Better TTS:** Swap `pyttsx3` → [Kokoro TTS](https://github.com/hexgrad/kokoro) (~82 MB, neural voices, ~1 s latency)
- **Better MT:** Swap Argos → ONNX-quantized MarianMT (Helsinki-NLP) for higher quality
- **VAD pre-filter:** Add Silero-VAD between `AudioCapture` and `ASREngine` to skip silence before it reaches Whisper, eliminating hallucinations entirely
- **More language pairs:** Run `python setup_models.py` after updating `TARGET_LANG` in `config.py`
