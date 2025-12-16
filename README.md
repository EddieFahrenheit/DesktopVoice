## DesktopVoice

Terminal app that listens for a wake word and (re)triggers Gemini’s “Use microphone” button in a Playwright-controlled browser window.

### Prereqs

- Python 3.10+
- A working microphone

System deps for `sounddevice` / PortAudio:

**Ubuntu**

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev
```

**macOS**

```bash
brew install portaudio
```

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
```

To use installed Google Chrome instead of Playwright Chromium, set `BROWSER_CHANNEL=chrome` in `.env`.

### Configure

Create a per-machine `.env`:

```bash
cp .env.example .env
```

Edit `.env`:

- `WAKEWORD`: wake word model name (e.g. `alexa`, `hey_jarvis`) or a local path to a custom `.onnx` model.
- `THRESH`: detection threshold (higher = fewer false positives).
- `COOLDOWN`: seconds to ignore repeat triggers after a detection.
- `MIC_RESTART`: `1` stops + restarts Gemini mic if already active; `0` leaves it running.

### First run (one-time)

1. Run the app (below) so the dedicated Playwright browser profile opens.
2. In that browser window, log into https://gemini.google.com/.
3. Select **Gemini 3 Pro** (if desired) and grant microphone permission when prompted.

The app uses a dedicated persistent profile directory (default: `.playwright_profile`) and does not reuse your main Chrome profile.

### Run

```bash
python app.py
```

Say your wake word; DesktopVoice will focus an existing Gemini tab (or open one) and click the microphone button. Press `Ctrl+C` to quit.

### Notes / troubleshooting

- If Playwright can’t find the mic button, Gemini UI selectors may have changed; check `app.py` selectors in `GeminiMicLauncher._start_selectors()`.
- Some openWakeWord models may download on first use; for fully offline use, point `WAKEWORD` at a local `.onnx` model file.
