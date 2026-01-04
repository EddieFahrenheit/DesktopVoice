## DesktopVoice

Terminal voice helper that listens for a wake word, transcribes your next phrase locally, then drives Google Chrome hands-free (Gemini / ChatGPT) by clicking the mic button.

Audio stays on-device for transcription. After the mic is clicked, your browser session behaves like normal (Gemini/ChatGPT may send audio to their servers, just like if you clicked the mic yourself).

### Prereqs

- Python 3.10+
- A working microphone
- Google Chrome installed

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
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: you do not need Playwright’s bundled Chromium if you use your installed Chrome (`BROWSER_CHANNEL=chrome`) or CDP mode (below).

### Configure

Create a per-machine `.env`:

```bash
cp .env.example .env
```

Edit `.env` (minimum):

- `WAKEWORD`: wake word model name (e.g. `alexa`, `hey_jarvis`) or a local path to a custom `.onnx` model.
- `THRESH`: detection threshold (higher = fewer false positives).
- `COOLDOWN`: seconds to ignore repeat triggers after a detection.
- `COMMAND_SECONDS`: how long to record after the wake word (e.g. `3.0`).
- `WHISPER_MODEL`: local speech-to-text model (e.g. `small`).

Chrome control mode (recommended: CDP + dedicated profile):

- `CHROME_CDP_URL`: set to `http://127.0.0.1:9222` to enable CDP mode.
- `CHROME_CDP_USER_DATA_DIR`: dedicated Chrome profile directory (example: `~/.desktopvoice_profile`).
- `CHROME_CDP_PROFILE_DIRECTORY`: usually `Default`.

Fallback (if you leave `CHROME_CDP_URL` blank):

- `BROWSER_CHANNEL=chrome` uses your installed Google Chrome
- `PROFILE_DIR` is the dedicated profile directory for Playwright-launched Chrome

### First run (one-time)

1. Start DesktopVoice (below).
2. Say your wake word, then say `open gemini`.
3. In the Chrome window that opens, log into https://gemini.google.com/ and allow microphone permission when prompted.

### Run

```bash
python -m desktopvoice
```

Say your wake word, then speak a command phrase:

- `open gemini`: open/focus Gemini and click the mic button
- `ask gemini`: assumes a Gemini tab already exists; focuses it and clicks mic (faster)
- `open chat`: open/focus ChatGPT and click the mic button

Press `Ctrl+C` to quit.

### Notes / troubleshooting

- If CDP mode doesn’t seem to work, check `curl http://127.0.0.1:9222/json/version`. If it fails, Chrome is not listening on the CDP port.
- Some openWakeWord models may download on first use; for fully offline use, point `WAKEWORD` at a local `.onnx` model file.
- If transcription is empty or errors, confirm `ffmpeg` is installed and try a smaller model (`WHISPER_MODEL=tiny`).
- If commands aren’t being recognized, matching is intentionally strict; edit `desktopvoice/commands.py` to add phrases.
