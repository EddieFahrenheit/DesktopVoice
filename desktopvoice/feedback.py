import shutil
import subprocess
import sys

def play_beep(start: bool) -> None:
    """
    Cross‑platform beep:
    - macOS: afplay
    - Ubuntu: paplay or aplay
    - fallback: terminal bell
    """
    try:
        if sys.platform == "darwin":
            sound = "/System/Library/Sounds/Ping.aiff" if start else "/System/Library/Sounds/Glass.aiff"
            subprocess.Popen(["afplay", sound], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return

        if shutil.which("paplay"):
            sound = "/usr/share/sounds/freedesktop/stereo/message.oga"
            subprocess.Popen(["paplay", sound], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return

        if shutil.which("aplay"):
            sound = "/usr/share/sounds/alsa/Front_Center.wav"
            subprocess.Popen(["aplay", sound], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
    except Exception:
        pass

    # Fallback (terminal bell)
    print("\a", end="", flush=True)
