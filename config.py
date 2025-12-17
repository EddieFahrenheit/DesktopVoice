import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    repo_dir: Path
    wakeword: str
    thresh: float
    cooldown_s: float

    mic_restart: bool
    browser_channel: str | None
    profile_dir: Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_config() -> AppConfig:
    repo_dir = Path(__file__).resolve().parent
    load_dotenv(repo_dir / ".env")

    wakeword = (os.getenv("WAKEWORD") or "").strip()
    thresh = float(os.getenv("THRESH", "0.6"))
    cooldown_s = float(os.getenv("COOLDOWN", "2.5"))

    mic_restart = _env_bool("MIC_RESTART", default=True)
    browser_channel = (os.getenv("BROWSER_CHANNEL") or "").strip() or None

    profile_dir_raw = (os.getenv("PROFILE_DIR") or ".playwright_profile").strip() or ".playwright_profile"
    profile_dir = Path(profile_dir_raw).expanduser()
    if not profile_dir.is_absolute():
        profile_dir = repo_dir / profile_dir

    return AppConfig(
        repo_dir=repo_dir,
        wakeword=wakeword,
        thresh=thresh,
        cooldown_s=cooldown_s,
        mic_restart=mic_restart,
        browser_channel=browser_channel,
        profile_dir=profile_dir,
    )

