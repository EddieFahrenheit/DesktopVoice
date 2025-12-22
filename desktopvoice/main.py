import os

from .audio_stream import MicAudioStream
from .config import load_config
from .stt import record_command_wav, transcribe_wav
from .wakeword import WakeWordListener

GEMINI_URL = "https://gemini.google.com/"
GEMINI_ORIGIN = "https://gemini.google.com"
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

    with MicAudioStream(sample_rate_hz=SAMPLE_RATE_HZ, frames_per_chunk=FRAMES_PER_CHUNK, channels=1, dtype="int16") as mic:
        while True:
            chunk = mic.read()[:, 0]  # mono
            best_name, best_score, triggered = listener.process(chunk)
            print(f"\rbest={best_name} score={best_score:.3f}  ", end="", flush=True)

            if triggered:
                print(f"\nDETECTED: {best_name} score={best_score:.3f}", flush=True)
                wav_path = record_command_wav(mic, sample_rate_hz=SAMPLE_RATE_HZ, seconds=cfg.command_seconds)
                try:
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

                # Cooldown should start *after* we've finished handling the wake-word event.
                # Otherwise, if recording+transcription takes longer than `cooldown_s`, we can
                # re-trigger immediately when we return to the wake-word loop.
                mic.drain()
                listener.mark_handled_now()
