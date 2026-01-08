## DesktopVoice

Desktop voice helper that listens for a wake word, transcribes your next phrase locally, then drives Google Chrome hands-free for Gemini or ChatGPT by clicking the mic / voice button.

Audio stays on-device for transcription. Once the mic is clicked, the browser session behaves like normal (Gemini/ChatGPT may send audio to their servers, just like if you clicked the mic yourself).

### What it does

- Wake word detection (openWakeWord)
- Local speech-to-text (faster-whisper)
- Simple command router (short phrases → actions)
- Chrome automation for Gemini / ChatGPT (Playwright + optional CDP)

### Prereqs

- Python 3.10/3.11 (code uses Python 3.10 syntax)
- A working microphone
- Google Chrome installed

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

Note: you do not need Playwright’s bundled Chromium if you use installed Chrome (`BROWSER_CHANNEL=chrome`) or CDP mode (below). If you do want Playwright’s Chromium, run:

```bash
python -m playwright install chromium
```

### Configure

Create a per-machine `.env`:

```bash
cp .env.example .env
```

Edit `.env` (minimum):

- `WAKEWORD`: wake word model name (e.g. `alexa`, `hey_jarvis`) or a local path to a custom `.onnx` model.
- `THRESH`: detection threshold (higher = fewer false positives).
- `COOLDOWN`: seconds to ignore repeat triggers after a detection.
- `COMMAND_SECONDS`: how long to record after the wake word (e.g. `1.5`).
- `WHISPER_MODEL`: local speech-to-text model (e.g. `small`).

Chrome control mode (recommended: CDP + dedicated profile):

- `CHROME_CDP_URL`: set to `http://127.0.0.1:9222` to enable CDP mode.
- `CHROME_CDP_USER_DATA_DIR`: dedicated Chrome profile directory (example: `~/.desktopvoice_profile`).
- `CHROME_CDP_PROFILE_DIRECTORY`: usually `Default`.

Fallback (if you leave `CHROME_CDP_URL` blank):

- `BROWSER_CHANNEL=chrome` uses your installed Google Chrome
- `PROFILE_DIR` is the dedicated profile directory for Playwright-launched Chrome

### Run

```bash
python -m desktopvoice
```

### First run (one-time)

1. Start DesktopVoice.
2. Say your wake word, then say `google` to open Gemini.
3. In the Chrome window, log in to https://gemini.google.com/ and allow microphone permission when prompted.
4. Optional: say `chat` once to open ChatGPT and log in at https://chatgpt.com/.

### Voice commands

Primary phrases (fast + reliable):

- `google`: open/focus Gemini
- `chat`: open/focus ChatGPT and start voice mode
- `voice`: click the mic / voice button on the most recent assistant tab
- `stop`: stop voice mode on the most recent assistant tab

There are additional alias phrases to reduce mis-hearings. See `desktopvoice/commands.py` to customize.

### CDP (Chrome DevTools Protocol) notes

If CDP doesn’t seem to work, check:

```bash
curl http://127.0.0.1:9222/json/version
```

If it fails, Chrome is not listening on the CDP port. Make sure Chrome is fully quit before starting it with CDP flags. A working macOS example is in `.env.example`.

### Troubleshooting

- **Slow STT:** use a smaller model (`WHISPER_MODEL=tiny` or `base`) and/or reduce `COMMAND_SECONDS`.
- **Mic button not found:** Gemini/ChatGPT labels change; update the patterns in `desktopvoice/browser.py`.
- **Google sign-in blocked by automation:** use CDP mode with a dedicated profile and log in once manually.
- **Command not recognized:** matching is intentionally strict; edit `desktopvoice/commands.py` to add aliases.
