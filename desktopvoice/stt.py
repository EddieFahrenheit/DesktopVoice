import os
import tempfile
import wave
from .audio_stream import MicAudioStream


def record_command_wav(mic: MicAudioStream, *, sample_rate_hz: int, seconds: float) -> str:
    """
    Record a short command from the microphone and save it as a temporary `.wav` file.

    Why WAV?
    - It's a simple, uncompressed container for raw PCM audio (in our case: 16kHz, mono, int16).
    - It's easy to write with Python's built-in `wave` module.
    - Speech-to-text libraries can reliably consume it.

    Returns the path to the temp WAV file.
    """
    chunks: list[bytes] = []
    frames_needed = int(sample_rate_hz * seconds)
    frames_got = 0

    print(f"Recording command for {seconds:.1f}s… speak now.", flush=True)
    while frames_got < frames_needed:
        chunk = mic.read()[:, 0]  # int16 mono
        chunks.append(chunk.tobytes())
        frames_got += len(chunk)

    fd, path = tempfile.mkstemp(suffix=".wav", prefix="desktopvoice_")
    os.close(fd)  # `mkstemp` returns an OS file descriptor; close it since `wave.open` will write the file.
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate_hz)
        wf.writeframes(b"".join(chunks))
    return path


def transcribe_wav(path: str, *, cfg) -> str:
    """
    Transcribe a WAV file to text using `faster-whisper` locally (no audio leaves the machine).

    Note: `faster-whisper` may require `ffmpeg` to be installed on your system, even for WAV input.
    """
    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError:
        print("Missing dependency: faster-whisper. Run `pip install -r requirements.txt`.")
        raise

    print(f"Transcribing with faster-whisper (model={cfg.whisper_model}, device={cfg.whisper_device})…", flush=True)
    model = WhisperModel(cfg.whisper_model, device=cfg.whisper_device, compute_type=cfg.whisper_compute_type)
    try:
        segments, _info = model.transcribe(path, beam_size=1, vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments).strip()
    except Exception as exc:
        print(f"Transcription failed: {exc}", flush=True)
        print("Tip: install ffmpeg (macOS: `brew install ffmpeg`, Ubuntu: `sudo apt-get install -y ffmpeg`).", flush=True)
        raise
    return text