import os

from .audio_stream import MicAudioStream
from .browser import BrowserController
from .commands import parse_command
from .config import load_config
from .feedback import play_beep, wake_display
from .hub_client import send_hub_command
from .stt import record_command_wav, transcribe_wav
from .wakeword import WakeWordListener

SAMPLE_RATE_HZ = 16000
CHUNK_SECONDS = 0.48
FRAMES_PER_CHUNK = int(SAMPLE_RATE_HZ * CHUNK_SECONDS)  # ~7680

def main():
    cfg = load_config()
    if not cfg.wakeword:
        print("Set WAKEWORD in .env (copy .env.example to .env).", flush=True)
        raise SystemExit(2)

    listener = WakeWordListener(
        wakeword=cfg.wakeword,
        thresh=cfg.thresh,
        cooldown_s=cfg.cooldown_s,
    )

    print(
        f"Listening… say the wake word. (thresh={cfg.thresh} cooldown={cfg.cooldown_s}s) Ctrl+C to stop.",
        flush=True,
    )

    # Lazy-start the browser so running the listener doesn't immediately open Chrome.
    browser: BrowserController | None = None
    try:
        with MicAudioStream(sample_rate_hz=SAMPLE_RATE_HZ, frames_per_chunk=FRAMES_PER_CHUNK, channels=1, dtype="int16") as mic:
            while True:
                chunk = mic.read()[:, 0]  # mono
                best_name, best_score, triggered = listener.process(chunk)
                print(f"\rbest={best_name} score={best_score:.3f}  ", end="", flush=True)

                if triggered:
                    wake_display()
                    play_beep(start=True)
                    print(f"\nDETECTED: {best_name} score={best_score:.3f}", flush=True)
                    try:
                        wav_path = record_command_wav(
                            mic, sample_rate_hz=SAMPLE_RATE_HZ, seconds=cfg.command_seconds
                        )
                        play_beep(start=False)
                        try:
                            text = transcribe_wav(wav_path, cfg=cfg)
                        finally:
                            try:
                                os.unlink(wav_path)
                            except OSError:
                                pass

                        if text:
                            print(f'Heard: "{text}"', flush=True)
                            command = parse_command(text)
                            if command is None:
                                if not cfg.hub_url:
                                    print("No matching command and HUB_URL not set.", flush=True)
                                else:
                                    ok, status, body = send_hub_command(
                                        hub_url=cfg.hub_url,
                                        text=text,
                                        api_key=cfg.hub_api_key,
                                        timeout_s=cfg.hub_timeout_s,
                                    )
                                    if ok:
                                        print(f"Hub OK: {body}", flush=True)
                                    elif status == 422:
                                        print(f"Hub no match: {body}", flush=True)
                                    else:
                                        print(f"Hub error ({status}): {body}", flush=True)
                            else:
                                if browser is None or not browser.ensure_ready():
                                    if browser is not None:
                                        browser.__exit__(None, None, None)
                                    browser = BrowserController(cfg)
                                    browser.__enter__()

                                print(f"Executing: {command}", flush=True)
                                try:
                                    if command == "open_gemini":
                                        browser.open_gemini()
                                    elif command == "open_chatgpt":
                                        browser.open_chatgpt_voice()
                                    elif command == "mic":
                                        browser.start_voice()
                                    else:
                                        ha_result = send_to_ha(text)
                                        print(
                                            "HA:",
                                            ha_result.get("response", {})
                                            .get("speech", {})
                                            .get("plain", {})
                                            .get("speech"),
                                        )
                                except Exception as exc:
                                    print(f"Browser action failed: {exc}", flush=True)
                                    # One recovery attempt
                                    browser.__exit__(None, None, None)
                                    browser = BrowserController(cfg)
                                    browser.__enter__()
                        else:
                            print('Heard: "" (no speech detected)', flush=True)
                    finally:
                        # Always reset cooldown and clear buffered audio.
                        mic.drain()
                        listener.mark_handled_now()

    finally:
        if browser is not None:
            browser.__exit__(None, None, None)
