import time
from pathlib import Path

try:
    from openwakeword.model import Model as WakeWordModel
    from openwakeword.utils import download_models
except ModuleNotFoundError:
    print("Missing dependency: openwakeword. Run `pip install -r requirements.txt`.")
    raise


class WakeWordListener:
    """
    Wraps openWakeWord wake-word detection and cooldown logic.

    This keeps `main.py` focused on orchestration (read audio -> detect -> record -> transcribe).
    """

    def __init__(self, *, wakeword: str, thresh: float, cooldown_s: float) -> None:
        self._wakeword = wakeword
        self._thresh = thresh
        self._cooldown_s = cooldown_s
        self._last_trigger = 0.0

        wakeword_path = Path(wakeword).expanduser()
        if wakeword_path.exists():
            wakeword_model_arg = str(wakeword_path)
        else:
            print("Downloading openWakeWord model files (first run only)…", flush=True)
            try:
                download_models(model_names=[wakeword])
            except Exception as exc:
                print(f"Failed to download openWakeWord model files: {exc}", flush=True)
                print(
                    "If you're offline, run again when you have internet, or set WAKEWORD to a local .onnx file path.",
                    flush=True,
                )
                raise
            wakeword_model_arg = wakeword

        print(f"Loading openWakeWord model: {wakeword}", flush=True)
        try:
            self._model = WakeWordModel(wakeword_models=[wakeword_model_arg], inference_framework="onnx")
        except Exception as exc:
            print(f"Failed to initialize openWakeWord model: {exc}", flush=True)
            print(
                "Tip: if you're using a custom model, set WAKEWORD to a local .onnx filepath in .env.",
                flush=True,
            )
            raise

    def process(self, chunk) -> tuple[str, float, bool]:
        preds = self._model.predict(chunk)

        if isinstance(preds, dict) and preds:
            best_name, best_score = max(preds.items(), key=lambda kv: float(kv[1]))
            best_score = float(best_score)
        else:
            best_name, best_score = "wakeword", float(preds) if preds is not None else 0.0

        now = time.time()
        triggered = best_score >= self._thresh and (now - self._last_trigger) >= self._cooldown_s
        if triggered:
            self._last_trigger = now

        return best_name, best_score, triggered

    def mark_handled_now(self) -> None:
        # Call this after record+transcribe so cooldown starts *after* handling.
        self._last_trigger = time.time()
