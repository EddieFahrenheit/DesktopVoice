import os

from .audio_stream import MicAudioStream
from .config import load_config
from .feedback import play_beep
from .hub_client import send_hub_command
from .stt import record_command_wav, transcribe_wav
from .wakeword import WakeWordListener

SAMPLE_RATE_HZ = 16000
CHUNK_SECONDS = 0.48
FRAMES_PER_CHUNK = int(SAMPLE_RATE_HZ * CHUNK_SECONDS)  # ~7680

ZERO_SHOT_ACTIONS = {
    "main_on": "main on",
    "kill_main": "main off",
    "bed_on": "bed on",
    "kill_bed": "bed off",
    "down_stairs_on": "downstairs on",
    "kill_down_stairs": "downstairs off",
    "wake_work_stay_shin": "work on",
    "kill_work_stay_shin": "work off",
    "wake_om_riht": "jarvis on",
    "kill_om_riht": "jarvis off",
}

def run_zero_shot_action(name: str, *, cfg) -> None:
    play_beep(start=True)
    text = ZERO_SHOT_ACTIONS.get(name)
    if not text:
        print(f"No zero-shot action configured for '{name}'.", flush=True)
        return
    if not cfg.hub_url:
        print("HUB_URL not set; cannot send zero-shot command.", flush=True)
        return
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

    command_listener = None
    if cfg.command_wakewords:
        command_listener = WakeWordListener(
            wakeword=list(cfg.command_wakewords),
            thresh=cfg.command_thresh,
            cooldown_s=cfg.command_cooldown_s,
        )


    print(
        f"Listening… say the wake word. (thresh={cfg.thresh} cooldown={cfg.cooldown_s}s) Ctrl+C to stop.",
        flush=True,
    )

    try:
        with MicAudioStream(sample_rate_hz=SAMPLE_RATE_HZ, frames_per_chunk=FRAMES_PER_CHUNK, channels=1, dtype="int16") as mic:
            while True:
                chunk = mic.read()[:, 0]  # mono
                best_name, best_score, triggered = listener.process(chunk)
                print(f"\rbest={best_name} score={best_score:.3f}  ", end="", flush=True)

                if command_listener is not None:
                    cmd_name, cmd_score, cmd_triggered = command_listener.process(chunk)
                    if cmd_triggered:
                        play_beep(start=True)
                        print(f"\nCOMMAND DETECTED: {cmd_name} score={cmd_score:.3f}", flush=True)
                        run_zero_shot_action(cmd_name, cfg=cfg)
                        mic.drain()
                        command_listener.mark_handled_now()
                        continue


                if triggered:
                    play_beep(start=True)
                    print(f"\nDETECTED: {best_name} score={best_score:.3f}", flush=True)
                    try:
                        wav_path = record_command_wav(
                            mic, sample_rate_hz=SAMPLE_RATE_HZ, seconds=cfg.command_seconds
                        )
                        try:
                            play_beep(start=False)
                            text = transcribe_wav(wav_path, cfg=cfg)
                        finally:
                            try:
                                os.unlink(wav_path)
                            except OSError:
                                pass

                        if text:
                            print(f'Heard: "{text}"', flush=True)
                        else:
                            print('Heard: "" (no speech detected)', flush=True)
                    finally:
                        # Always reset cooldown and clear buffered audio.
                        mic.drain()
                        listener.mark_handled_now()
    finally:
        print("\nExiting, bye!", flush=True)