import shutil
import subprocess
import sys

def play_beep(start: bool) -> None:
    """
    Cross‑platform beep:
    - macOS: afplay
    - Ubuntu: paplay
    - fallback: terminal bell
    """
    try:
        if sys.platform == "darwin":
            sound = "/System/Library/Sounds/Ping.aiff" if start else "/System/Library/Sounds/Glass.aiff"
            subprocess.Popen(["afplay", sound], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return

        if shutil.which("paplay"):
            sound = "/usr/share/sounds/Yaru/stereo/device-added.oga" if start else "/usr/share/sounds/Yaru/stereo/device-removed.oga"
            subprocess.Popen(["paplay", sound], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return

    except Exception:
        pass

    # Fallback (terminal bell)
    print("\a", end="", flush=True)