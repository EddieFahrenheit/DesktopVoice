import queue
from typing import Any

import numpy as np
import sounddevice as sd


class MicAudioStream:
    def __init__(
        self,
        *,
        sample_rate_hz: int,
        frames_per_chunk: int,
        channels: int = 1,
        dtype: str = "int16",
        queue_size: int = 8,
    ) -> None:
        self._sample_rate_hz = sample_rate_hz
        self._frames_per_chunk = frames_per_chunk
        self._channels = channels
        self._dtype = dtype

        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=queue_size)
        self._printed_frames = False
        self._stream: sd.InputStream | None = None

    def __enter__(self) -> "MicAudioStream":
        self._stream = sd.InputStream(
            samplerate=self._sample_rate_hz,
            channels=self._channels,
            dtype=self._dtype,
            blocksize=self._frames_per_chunk,
            callback=self._callback,
        )
        self._stream.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool | None:
        assert self._stream is not None
        return self._stream.__exit__(exc_type, exc, tb)

    def read(self) -> np.ndarray:
        return self._q.get()

    def drain(self) -> int:
        """
        Discard any already-buffered audio chunks.

        This is useful after a wake word is detected (and we record a command) so we don't
        immediately re-process stale audio that accumulated in the queue while we were busy
        transcribing.
        """
        dropped = 0
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
            else:
                dropped += 1
        return dropped

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:  # noqa: ARG002
        if status:
            print(f"\nAudio status: {status}", flush=True)

        if not self._printed_frames:
            print(f"Callback frames per block: {frames}", flush=True)
            self._printed_frames = True

        try:
            self._q.put_nowait(indata.copy())
        except queue.Full:
            pass
