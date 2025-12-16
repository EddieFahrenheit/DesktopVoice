from dotenv import load_dotenv
import os
import queue
import time
import numpy as np
import sounddevice as sd

load_dotenv()

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
    print("Starting mic stream… Ctrl+C to stop.", flush=True)
    with sd.InputStream(
        samplerate=SAMPLE_RATE_HZ,
        channels=1,
        dtype="float32",          # easier for level math; values ~[-1.0, 1.0]
        blocksize=FRAMES_PER_CHUNK,
        callback=callback,
    ):
        while True:
            chunk = q.get()
            chunk = chunk[:, 0]  # mono
            rms = float(np.sqrt(np.mean(chunk * chunk)))
            peak = float(np.max(np.abs(chunk)))
            bar = "#" * min(50, int(peak * 100))
            print(f"\rRMS={rms:0.4f}  PEAK={peak:0.4f}  {bar:<50}", end="", flush=True)
            time.sleep(0.01)


if __name__ == "__main__":
    main()