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
    chrome_cdp_url: str | None
    chrome_cdp_user_data_dir: Path
    chrome_cdp_profile_directory: str

    # Faster Whisper settings
    command_seconds: float
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_config() -> AppConfig:
    repo_dir = Path(__file__).resolve().parents[1]  # repo root
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

    # Optional: attach to an already-running Chrome via the Chrome DevTools Protocol (CDP).
    # Example: CHROME_CDP_URL=http://127.0.0.1:9222
    chrome_cdp_url = (os.getenv("CHROME_CDP_URL") or "").strip() or None
    chrome_cdp_user_data_dir_raw = (os.getenv("CHROME_CDP_USER_DATA_DIR") or "").strip() or None
    if chrome_cdp_user_data_dir_raw:
        chrome_cdp_user_data_dir = Path(chrome_cdp_user_data_dir_raw).expanduser()
        if not chrome_cdp_user_data_dir.is_absolute():
            chrome_cdp_user_data_dir = repo_dir / chrome_cdp_user_data_dir
    else:
        # Default to the same profile directory we use when launching via Playwright.
        chrome_cdp_user_data_dir = profile_dir

    chrome_cdp_profile_directory = (os.getenv("CHROME_CDP_PROFILE_DIRECTORY") or "").strip() or "Default"

    # Load Faster Whisper settings
    command_seconds = float(os.getenv("COMMAND_SECONDS", "3.0"))
    whisper_model = (os.getenv("WHISPER_MODEL") or "small").strip()
    whisper_device = (os.getenv("WHISPER_DEVICE") or "cpu").strip()
    whisper_compute_type = (os.getenv("WHISPER_COMPUTE_TYPE") or "int8").strip()

    return AppConfig(
        repo_dir=repo_dir,
        wakeword=wakeword,
        thresh=thresh,
        cooldown_s=cooldown_s,
        mic_restart=mic_restart,
        browser_channel=browser_channel,
        profile_dir=profile_dir,
        chrome_cdp_url=chrome_cdp_url,
        chrome_cdp_user_data_dir=chrome_cdp_user_data_dir,
        chrome_cdp_profile_directory=chrome_cdp_profile_directory,
        command_seconds=command_seconds,
        whisper_model=whisper_model,
        whisper_device=whisper_device,
        whisper_compute_type=whisper_compute_type,
    )
