# PolyglotTalk Dashboard

Real-time browser dashboard for the PolyglotTalk offline speech-to-speech pipeline.
Connects over WebSocket to a FastAPI backend that bridges the 4-thread Python pipeline.

```
Microphone → [ASR] → [Translator] → [TTS]
                 ↘                ↘         ↘
              ws broadcast    ws broadcast   ws broadcast + /audio/{file}
                              ↗
                   Browser dashboard (Vite + React + Tailwind)
```

---

## Prerequisites

| Tool | Version |
|---|---|
| Python | 3.11.x (same venv as the pipeline) |
| Node.js | 18+ |
| npm | 9+ |

Backend deps (already installed if you ran `uv pip install -r requirements.txt`):
```
fastapi, uvicorn[standard], websockets
```

If not, install manually:
```pwsh
uv pip install fastapi "uvicorn[standard]" websockets
```

---

## Running the Dashboard

### Terminal 1 — Python pipeline with dashboard server

```pwsh
# Activate your venv first
.venv\Scripts\activate

# Start pipeline + dashboard WebSocket server (port 8765)
python main.py --dashboard

# Optional: choose target language and custom port
python main.py --dashboard --target tam --dashboard-port 8765
```

You'll see:
```
============================================================
 PolyglotTalk v0.1 — Offline Speech-to-Speech Translation
 EN → HIN  |  TTS: MMS-TTS (cuda)  |  No cloud APIs
  Dashboard: http://localhost:8765  (WS: ws://localhost:8765/ws)
============================================================
✓ Pipeline ready. Speak now…
```

### Terminal 2 — React frontend dev server

```pwsh
cd dashboard
npm install       # first time only
npm run dev
```

Then open **http://localhost:5173** in your browser.

> The React app connects to `ws://localhost:8765/ws` automatically.
> If the connection drops it auto-reconnects every 2 seconds.

---

## Dashboard Layout

```
┌─────────────────────────────────────────────┬──────────────┐
│  Live Transcription                         │              │
│  (growing partial text + blinking cursor)   │  Audio Files │
│─────────────────────────────────────────────│  (WAV player)│
│  Translation (→HIN / →TAM / etc.)           │              │
│  (fades in on each new translation)         │──────────────│
│                                             │  Event Log   │
│                                             │  (chunk log) │
├─────────────────────────────────────────────┴──────────────┤
│  Stats: WS · Mic · Chunks · Sentences · Avg Latency · TTS  │
└────────────────────────────────────────────────────────────┘
```

### Panels

| Panel | Description |
|---|---|
| **Live Transcription** | ASR chunks accumulate word-by-word with a blinking cursor. When a sentence is flushed it fades out. Older confirmed sentences appear muted above. |
| **Translation** | Latest translated text fades in on each `translation_done` event. Supports RTL scripts (Gujarati, Hindi, etc.) via `dir="auto"`. |
| **Audio Files** | Each saved `chunk_NNNN.wav` appears as a playable row. Click ▶ to play audio directly in the browser via the `/audio/` API endpoint. |
| **Event Log** | Colour-coded scrolling log of every pipeline event: `ASR` (blue), `SENT` (purple), `TRANS` (amber), `TTS` (green). |
| **Stats Bar** | Chunk count, sentence count, translation count, TTS file count, rolling average end-to-end latency (ms), mic status. |

---

## WebSocket Events

All events are JSON objects received on `ws://localhost:8765/ws`.

| Type | Fields | Description |
|---|---|---|
| `asr_chunk` | `chunk_id`, `text` | Raw deduplicated ASR fragment |
| `sentence_flushed` | `chunk_id`, `text` | Complete sentence flushed from buffer |
| `translation_done` | `chunk_id`, `text`, `lang` | Translated text (ISO 639-3 `lang`) |
| `tts_saved` | `chunk_id`, `filename`, `latency_ms` | WAV file written; `latency_ms` = capture→file |
| `pipeline_status` | `status` | `"ready"` or `"stopped"` |
| `connected` | — | Sent once on WebSocket handshake |

All events also carry a `ts` field (Unix timestamp seconds).

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `WS` | `/ws` | WebSocket event stream |
| `GET` | `/audio/{filename}` | Serve `output/chunk_NNNN.wav` |
| `GET` | `/health` | `{"status": "ok"}` |

---

## Ports

| Service | Port |
|---|---|
| FastAPI / WebSocket server | `8765` (configurable via `--dashboard-port`) |
| Vite React dev server | `5173` |

---

## Production Build (optional)

```pwsh
cd dashboard
npm run build
# Serve from dashboard/dist/ with any static server
```

For production you can point FastAPI to serve `dashboard/dist/` as static files instead of redirecting to port 5173.
