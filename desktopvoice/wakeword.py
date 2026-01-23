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

    def __init__(self, *, wakeword: str | list[str], thresh: float, cooldown_s: float) -> None:
        self._thresh = thresh
        self._cooldown_s = cooldown_s
        self._last_trigger = 0.0

        if isinstance(wakeword, str):
            wakewords = [wakeword]
        else:
            wakewords = list(wakeword)

        if not wakewords:
            raise ValueError("wakeword list is empty.")
        self._wakewords = wakewords

        self._name_map: dict[str, str] = {}
        model_args: list[str] = []
        for entry in wakewords:
            label = Path(entry).stem
            self._name_map[entry] = label
            self._name_map[Path(entry).name] = label
            self._name_map[Path(entry).stem] = label

            path = Path(entry).expanduser()
            looks_like_path = path.suffix.lower() == ".onnx" or "/" in entry or "\\" in entry
            if path.exists():
                model_args.append(str(path))
                self._name_map[str(path)] = label
                continue

            if looks_like_path:
                raise FileNotFoundError(f"Wakeword model not found: {path}")

            print("Downloading openWakeWord model files (first run only)...", flush=True)
            try:
                download_models(model_names=[entry])
            except Exception as exc:
                print(f"Failed to download openWakeWord model files: {exc}", flush=True)
                print(
                    "If you're offline, run again when you have internet, or set WAKEWORD to a local .onnx file path.",
                    flush=True,
                )
                raise
            model_args.append(entry)

        print(f"Loading openWakeWord model(s): {', '.join(wakewords)}", flush=True)
        try:
            self._model = WakeWordModel(wakeword_models=model_args, inference_framework="onnx")
        except Exception as exc:
            print(f"Failed to initialize openWakeWord model: {exc}", flush=True)
            print(
                "Tip: if you're using a custom model, set WAKEWORD to a local .onnx filepath in .env.",
                flush=True,
            )
            raise

    def _label_for(self, key: str) -> str:
        if key in self._name_map:
            return self._name_map[key]
        stem = Path(key).stem
        return self._name_map.get(stem, key)

    def process(self, chunk) -> tuple[str, float, bool]:
        preds = self._model.predict(chunk)

        if isinstance(preds, dict) and preds:
            best_key, best_score = max(preds.items(), key=lambda kv: float(kv[1]))
            best_score = float(best_score)
            best_name = self._label_for(best_key)
        else:
            best_name = self._label_for(self._wakewords[0]) if self._wakewords else "wakeword"
            best_score = float(preds) if preds is not None else 0.0

        now = time.time()
        triggered = best_score >= self._thresh and (now - self._last_trigger) >= self._cooldown_s
        if triggered:
            self._last_trigger = now

        return best_name, best_score, triggered

    def mark_handled_now(self) -> None:
        # Call this after record+transcribe so cooldown starts after handling.
        self._last_trigger = time.time()

