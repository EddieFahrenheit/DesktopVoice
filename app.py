import os
import queue
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

try:
    from openwakeword.model import Model as WakeWordModel
    from openwakeword.utils import download_models
except ModuleNotFoundError:
    print("Missing dependency: openwakeword. Run `pip install -r requirements.txt`.")
    raise

SAMPLE_RATE_HZ = 16000
CHUNK_SECONDS = 0.48
FRAMES_PER_CHUNK = int(SAMPLE_RATE_HZ * CHUNK_SECONDS)  # ~7680

q = queue.Queue(maxsize=8)
printed_frames = False


def callback(indata, frames, time_info, status):  # noqa: ARG001
    global printed_frames
    if status:
        print(f"\nAudio status: {status}", flush=True)

    if not printed_frames:
        print(f"Callback frames per block: {frames}", flush=True)
        printed_frames = True

    try:
        q.put_nowait(indata.copy())
    except queue.Full:
        pass


def main():
    repo_dir = Path(__file__).resolve().parent
    load_dotenv(repo_dir / ".env")

    wakeword = (os.getenv("WAKEWORD") or "").strip()
    if not wakeword:
        print("Set WAKEWORD in .env (copy .env.example to .env).", flush=True)
        raise SystemExit(2)

    thresh = float(os.getenv("THRESH", "0.6"))
    cooldown_s = float(os.getenv("COOLDOWN", "2.5"))

    wakeword_path = Path(wakeword).expanduser()
    if wakeword_path.exists():
        wakeword_model_arg = str(wakeword_path)
    else:
        wakeword_model_arg = wakeword
        print("Downloading openWakeWord model files (first run only)…", flush=True)
        try:
            download_models(model_names=[wakeword])
        except Exception as exc:
            print(f"Failed to download openWakeWord model files: {exc}", flush=True)
            print(
                "If you're offline, run again when you have internet, or set WAKEWORD to a local .onnx file path.",
                flush=True,
            )
            raise SystemExit(1)

    print(f"Loading openWakeWord model: {wakeword}", flush=True)
    try:
        model = WakeWordModel(wakeword_models=[wakeword_model_arg], inference_framework="onnx")
    except Exception as exc:
        print(f"Failed to initialize openWakeWord model: {exc}", flush=True)
        print(
            "Tip: if you're using a custom model, set WAKEWORD to a local .onnx filepath in .env.",
            flush=True,
        )
        raise SystemExit(1)

    print(
        f"Listening… say the wake word. (thresh={thresh} cooldown={cooldown_s}s) Ctrl+C to stop.",
        flush=True,
    )

    last_trigger = 0.0
    with sd.InputStream(
        samplerate=SAMPLE_RATE_HZ,
        channels=1,
        dtype="int16",
        blocksize=FRAMES_PER_CHUNK,
        callback=callback,
    ):
        while True:
            chunk = q.get()[:, 0]  # mono
            preds = model.predict(chunk)

            if isinstance(preds, dict) and preds:
                best_name, best_score = max(preds.items(), key=lambda kv: float(kv[1]))
                best_score = float(best_score)
            else:
                best_name, best_score = "wakeword", float(preds) if preds is not None else 0.0

            print(f"\rbest={best_name} score={best_score:.3f}  ", end="", flush=True)

            now = time.time()
            if best_score >= thresh and (now - last_trigger) >= cooldown_s:
                last_trigger = now
                print(f"\nDETECTED: {best_name} score={best_score:.3f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.", flush=True)
