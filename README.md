# PolyglotTalk

Real-time, fully **offline** Speech-to-Speech Translation (S2ST) running entirely on CPU — no cloud APIs, no GPU required.

```
Microphone → [Whisper ASR] → [Argos Translate] → [IndicF5 TTS] → WAV Files
```

Speak English into your microphone; translated speech is synthesised and saved to `output/chunk_*.wav` files (preventing mic feedback). Each stage runs in its own thread so recording, transcribing, translating, and synthesis all happen **simultaneously**.

---

## Features

- **Fully offline** — all models run locally after a one-time download
- **CPU-only** — no GPU or CUDA required; works on any modern laptop
- **True pipeline parallelism** — 4 threads run concurrently with ~4 s end-to-end latency
- **No mic feedback** — TTS output saved to files, not played through speakers
- **Auto-stop** — pipeline exits cleanly after 2.5 s of silence
- **Live console output** — transcription, translation, and TTS file paths printed as they happen
- **Configurable** — source/target language, TTS speed, and output directory via CLI flags
- **Hallucination filtering** — common Whisper silence-artifacts ("Thank you", "Bye") are blocked

---

## Demo Output

```
============================================================
 PolyglotTalk v0.1 — Offline Speech-to-Speech Translation
 EN → HI  |  TTS: IndicF5 (cpu)  |  No cloud APIs
 TTS output saved to: output/chunk_NNNN.wav
============================================================
✓ Pipeline ready. Speak now… (Ctrl+C to stop)
[ASR   #1] Hello, how are you today?
[→HI  #1] नमस्ते, आप आज कैसे हैं?
[TTS  #1] saved → output/chunk_0001.wav
[ASR   #2] I'm doing well, thank you.
[→HI  #2] मैं ठीक हूँ, धन्यवाद।
[TTS  #2] saved → output/chunk_0002.wav
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
sudo apt install -y libportaudio2 portaudio19-dev pulseaudio libpulse0 alsa-utils
```

| Package | Purpose |
|---|---|
| `libportaudio2` | PortAudio C library — required by `sounddevice` |
| `pulseaudio` / `libpulse0` | Audio routing (WSL2: connects to WSLg's RDP microphone) |
| `alsa-utils` | ALSA audio utilities — needed for audio device enumeration |

### HuggingFace CLI (for accessing gated models)

IndicF5 is a **gated repository** requiring authentication. Install the HF CLI and log in:

```bash
# Install HuggingFace CLI
curl -LsSf https://hf.co/cli/install.sh | bash

# Log in (creates ~/.cache/huggingface/token)
hf auth login
```

When prompted, paste your [HuggingFace API token](https://huggingface.co/settings/tokens).

**Before running `setup_models.py`:** Visit https://huggingface.co/ai4bharat/IndicF5 and click "Access repository" to accept the model's terms.

> **WSL2 users:** Audio is bridged via WSLg on Windows 11. After installing
> `libpulse0`, verify your mic appears with:
> ```bash
> PULSE_SERVER=unix:/mnt/wslg/PulseServer pactl list sources short
> ```
> You should see `RDPSource` listed. If not, run `wsl --update` from PowerShell.

> **TTS output files:** Synthesised speech is saved to `output/chunk_NNNN.wav`
> files (where N is the chunk ID). This avoids microphone feedback during
> live translation and uses IndicF5, a high-quality neural TTS model for
> Indian languages, running fully on your CPU (or GPU if available).

---

## Quick Start

### 1. Create environment and install Python dependencies

```bash
# Using uv (recommended)
uv venv --python 3.11.9
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Download models (one-time, requires internet, ~2.5 GB)

```bash
python setup_models.py
```

This downloads:
- `faster-whisper` `base.en` int8 model (~150 MB) → `~/.cache/huggingface/hub/`
- Argos Translate `en→hi` language pack (~100 MB) → `~/.local/share/argos-translate/`
- AI4Bharat IndicF5 TTS model (~2 GB, cached by transformers) → `~/.cache/huggingface/hub/`
- Hindi reference audio prompt for voice cloning (~1 MB) → `prompts/`

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
| `--log-level LEVEL` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Examples

```bash
# English → Spanish
python main.py --source en --target es

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
├── tts_engine.py        # Thread 4: TranslatedSegment → WAV files (IndicF5)
├── models.py            # Dataclasses: AudioChunk, TextSegment, TranslatedSegment
├── setup_models.py      # One-time model download + smoke-test script
├── requirements.txt     # Pinned Python dependencies
├── IMPLEMENTATION_PLAN.md
└── tests/
    ├── test_audio_capture.py   # Live mic recording test
    ├── test_asr.py             # Whisper transcription test
    ├── test_translator.py      # Argos translation test
    ├── test_tts.py             # IndicF5 synthesis test
    ├── test_context.py         # Context-continuity unit tests (mocked)
    └── test_pipeline_e2e.py    # Full pipeline integration test
```

---

## How It Works

```
┌──────────────┐  audio_queue  ┌──────────────┐  text_queue  ┌──────────────┐  tts_queue  ┌──────────────┐
│ AudioCapture │  (maxsize=2)  │  ASREngine   │  (maxsize=2) │  Translator  │ (maxsize=2) │  TTSEngine   │
│  Thread 1    │ ───────────►  │  Thread 2    │ ───────────► │  Thread 3    │ ──────────► │  Thread 4    │
│  sounddevice │  AudioChunk   │faster-whisper│  TextSegment │argos-translate  TranslatedSeg│  IndicF5     │
└──────────────┘               └──────────────┘              └──────────────┘              └──────────────┘
```

All four threads run simultaneously. At steady state:

- **Thread 1** records the next 2.5-second buffer while Thread 2 is still transcribing the current one
- **Thread 2** transcribes while Thread 3 translates the previous chunk
- **Thread 3** translates while Thread 4 synthesises and saves the chunk before that

End-to-end latency ≈ `chunk_duration + ASR_time + MT_time` ≈ **4 seconds** — not the sum of all stage durations.

**Backpressure:** All queues use a drop-oldest strategy — if a downstream stage falls behind, the oldest unprocessed item is evicted to make room for the freshest one, so the pipeline always stays current.

**Context continuity:** The translator maintains a rolling window of the last 2 source segments, prepending them as context to each new translation to reduce sentence-boundary errors.

**Audio isolation:** TTS output is saved to `output/chunk_NNNN.wav` files instead of being played through speakers. This prevents synthesised speech from feeding back into the microphone, which would corrupt future ASR and create feedback loops.

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
| `TTS_OUTPUT_DIR` | `"output"` | Directory where synthesised WAV files are saved |
| `INDICF5_MODEL_ID` | `"ai4bharat/IndicF5"` | HF model ID for IndicF5 TTS |
| `INDICF5_DEVICE` | `"auto"` | Device for IndicF5 (`"auto"`, `"cuda"`, or `"cpu"`) |
| `INDICF5_REF_AUDIO_PATH` | `"prompts/HIN_F_HAPPY_00001.wav"` | Path to reference audio for voice cloning |
| `INDICF5_REF_TEXT` | `""` | Transcript of reference audio (empty = auto-transcribe) |

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
| `test_asr.py` | LibriSpeech dev-clean dataset | WER evaluation on 100 utterances; skipped if dataset absent |
| `test_translator.py` | Models installed | Verifies Devanagari output |
| `test_tts.py` | IndicF5 reference audio | Checks 24 kHz WAV output; skipped if reference audio missing |
| `test_context.py` | Nothing | Fully mocked — runs anywhere |
| `test_pipeline_e2e.py` | Models installed | Uses synthetic audio, no mic needed |

---

## Dependencies

```
faster-whisper==1.1.0       # ASR — CTranslate2-optimised Whisper
argostranslate==1.9.6       # Machine translation — OpenNMT + CTranslate2
f5-tts @ git+https://...    # IndicF5 TTS — high-quality neural TTS for Indian languages
sounddevice==0.5.1          # Microphone input via PortAudio
soundfile==0.13.1           # FLAC/WAV I/O for ASR benchmarking + TTS output
jiwer==4.0.0                # WER calculation for ASR evaluation
numpy>=1.24,<2.0            # Audio arrays
scipy>=1.11                 # Signal processing utilities
pytest==8.3.4               # Testing
transformers==5.2.0         # AutoModel for IndicF5 loading
torch==2.10.0               # PyTorch (CPU ok, CUDA optional)
```

---

## Future Upgrades

- **More language pairs:** Run `python setup_models.py` after updating `TARGET_LANG` in `config.py`
- **Live playback:** Optional mode to play TTS output to speakers (separate input device to prevent feedback)
- **Improved VAD:** Add Silero-VAD before `ASREngine` to pre-filter silence, eliminating ASR hallucinations
- **Custom voice cloning:** Provide your own Hindi reference audio in `prompts/` directory for personalized TTS voice
