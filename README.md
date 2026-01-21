## DesktopVoice

Desktop voice helper that listens for a wake word, records a short command, transcribes it locally, then either:
- drives Gemini/ChatGPT in Chrome by clicking the mic or voice button, or
- forwards unmatched text to an optional Hub API for Home Assistant.

Audio stays on-device for wake word detection and transcription. Once the mic is clicked, the browser session behaves like normal (Gemini/ChatGPT may send audio to their servers, just like if you clicked the mic yourself).

### What it does

- Wake word detection (openWakeWord)
- Local speech-to-text (faster-whisper)
- Command routing (exact phrase match + optional Hub forwarding)
- Chrome automation for Gemini / ChatGPT (Playwright, optional CDP)
- Optional Hub API for Home Assistant control (FastAPI + MCP)

### Prereqs

- Python 3.10/3.11
- A working microphone
- Google Chrome installed (for CDP or `BROWSER_CHANNEL=chrome`)
- `ffmpeg` recommended for faster-whisper

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

If you plan to use Playwright's bundled Chromium (default when `BROWSER_CHANNEL` is unset), install it:

```bash
python -m playwright install chromium
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

Chrome control:

- `CHROME_CDP_URL`: set to `http://127.0.0.1:9222` to enable CDP mode.
- `CHROME_CDP_USER_DATA_DIR`: dedicated Chrome profile directory for CDP (defaults to `PROFILE_DIR`).
- `CHROME_CDP_PROFILE_DIRECTORY`: usually `Default`.
- `BROWSER_CHANNEL=chrome`: use your installed Google Chrome (otherwise Playwright's Chromium).
- `PROFILE_DIR`: persistent profile directory for Playwright-launched Chrome.

Hub forwarding (optional):

- `HUB_URL`: Hub API base URL (unmatched commands are POSTed to `/hub/command`).
- `HUB_API_KEY`: optional header sent as `X-API-Key` (server does not enforce by default).
- `HUB_TIMEOUT`: request timeout in seconds.

Home Assistant (Hub API host only):

- `HA_URL`: Home Assistant base URL (example: `http://192.168.122.195:8123`).
- `HA_TOKEN`: long-lived access token.
- `HA_LANGUAGE`: language for HA Assist (default `en`).

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
curl -X POST http://192.168.1.160:8000/hub/command \
  -H "Content-Type: application/json" \
  -d '{"text":"turn on jarvis"}'
```

Customize hub routing in `desktopvoice/hub_routes.py` (keywords, scripts, lights, switches).

### First run (one-time)

1. Start DesktopVoice.
2. Say your wake word, then say `google` (or `chat`).
3. In the Chrome window, log in to https://gemini.google.com/ or https://chatgpt.com/ and allow microphone permission.
4. After that, say `voice` to click the mic button on the most recent assistant tab.

### Voice commands

Primary phrases (fast + reliable):

- `google`: open/focus Gemini
- `chat`: open/focus ChatGPT and start voice mode
- `voice` or `mic`: click the mic / voice button on the most recent assistant tab

There are additional alias phrases to reduce mis-hearings. See `desktopvoice/commands.py` to customize.

### CDP (Chrome DevTools Protocol) notes

If CDP doesn't seem to work, check:

```bash
curl http://127.0.0.1:9222/json/version
```

If it fails, Chrome is not listening on the CDP port. Make sure Chrome is fully quit before starting it with CDP flags. A working macOS example is in `.env.example`.

### Troubleshooting

- **Slow STT:** use a smaller model (`WHISPER_MODEL=tiny` or `base`) and/or reduce `COMMAND_SECONDS`.
- **Wake word not triggering:** lower `THRESH`, check microphone input, or set `WAKEWORD` to a local `.onnx` file.
- **Mic button not found:** Gemini/ChatGPT labels change; update the patterns in `desktopvoice/browser.py`.
- **Google sign-in blocked by automation:** use CDP mode with a dedicated profile and log in once manually.
- **Hub errors:** verify `HA_URL`, `HA_TOKEN`, and that `uvx ha-mcp` runs on the Hub host.
