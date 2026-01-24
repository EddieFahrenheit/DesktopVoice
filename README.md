## DesktopVoice

Desktop voice helper that listens for a primary wake word, records a short command, and transcribes locally. It also supports zero-shot command wakewords (ONNX models) that map directly to Home Assistant actions via an optional Hub API.

Audio stays on-device for wake word detection and transcription.

### What it does

- Wake word detection (openWakeWord)
- Optional zero-shot command wakewords (ONNX) -> Hub action
- Local speech-to-text (faster-whisper)
- Optional Hub API for Home Assistant control (FastAPI + MCP)

### Prereqs

- Python 3.10/3.11
- A working microphone
- `ffmpeg` recommended for faster-whisper
- PortAudio for `sounddevice`

**macOS: upgrade Python if needed**

Check your Python version:

```bash
python3 --version
```

If it's below 3.10, install a newer Python with Homebrew:

```bash
brew install python@3.11
python3.11 --version
```

System deps:

- `sounddevice` needs PortAudio
- `faster-whisper` works best with `ffmpeg` installed

**Ubuntu**

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev ffmpeg
```

**macOS**

```bash
brew install portaudio ffmpeg
```

### Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Configure

Create a per-machine `.env`:

```bash
cp .env.example .env
```

Edit `.env` (see `.env.example` for the full list):

Core settings:

- `WAKEWORD`: model name (e.g. `hey_mycroft`, `hey_jarvis`) or a local `.onnx` path.
- `THRESH`: detection threshold (higher = fewer false positives).
- `COOLDOWN`: seconds to ignore repeat triggers after a detection.
- `COMMAND_SECONDS`: how long to record after the wake word.
- `WHISPER_MODEL`: faster-whisper model (e.g. `tiny`, `base`, `small`).
- `WHISPER_DEVICE`: `cpu` or `cuda` (if supported).
- `WHISPER_COMPUTE_TYPE`: e.g. `int8`, `float16`.

Zero-shot command wakewords:

- `COMMAND_WAKEWORDS`: comma-separated list of ONNX paths for command models.
- `COMMAND_THRESH`: detection threshold for command models.
- `COMMAND_COOLDOWN`: cooldown seconds for command models.

Hub forwarding (optional):

- `HUB_URL`: Hub API base URL (actions are POSTed to `/hub/action`).
- `HUB_API_KEY`: optional header sent as `X-API-Key` (server does not enforce by default).
- `HUB_TIMEOUT`: request timeout in seconds.

Home Assistant (Hub API host only):

- `HA_URL`: Home Assistant base URL (example: `http://192.168.122.195:8123`).
- `HA_TOKEN`: long-lived access token.
- `HA_LANGUAGE`: language for HA Assist (default `en`).

### Zero-shot model setup

1. Put ONNX models in `models/` (or any path you prefer).
2. Set `COMMAND_WAKEWORDS` to those model paths.
3. Ensure each model's filename stem matches an entry in `desktopvoice/hub_routes.py` `ACTION_MAP`.
4. Keep `ZERO_SHOT_ACTIONS` in `desktopvoice/main.py` aligned with `ACTION_MAP` (if you use the local allowlist).

Example:

- `models/main_on.onnx` -> action key `main_on`
- `ACTION_MAP["main_on"] = {"domain": "light", "service": "turn_on", ...}`

### Run

```bash
source .venv/bin/activate
python -m desktopvoice
```

### Run Hub API (optional)

Requirements:

- `uvx` available on the Hub host (install via `pipx install uv` or `pip install uv`).
- `ha-mcp` available to `uvx` (used by the MCP client to talk to Home Assistant).

Start the API server:

```bash
uvicorn desktopvoice.hub:app --host 0.0.0.0 --port 8000 --reload
```

Smoke test:

```bash
curl http://192.168.1.160:8000/hub/health
curl -X POST http://192.168.1.160:8000/hub/action \
  -H "Content-Type: application/json" \
  -d '{"action":"main_on"}'
```

Customize `ACTION_MAP` in `desktopvoice/hub_routes.py` to match your scripts and entities.

### Troubleshooting

- **Slow STT:** use a smaller model (`WHISPER_MODEL=tiny` or `base`) and/or reduce `COMMAND_SECONDS`.
- **Wake word not triggering:** lower `THRESH`, check microphone input, or set `WAKEWORD` to a local `.onnx` file.
- **Zero-shot action not firing:** confirm the ONNX filename stem matches `ACTION_MAP` and the entry exists.
- **Hub errors:** verify `HA_URL`, `HA_TOKEN`, and that `uvx ha-mcp` runs on the Hub host.
