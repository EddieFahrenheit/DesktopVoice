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

    command_wakewords: tuple[str, ...]
    command_thresh: float
    command_cooldown_s: float


    ha_url: str | None
    ha_token: str | None
    ha_language: str

    hub_url: str | None
    hub_api_key: str | None
    hub_timeout_s: float

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

def _normalize_wakeword_ref(value: str, repo_dir: Path) -> str:
    value = value.strip()
    if not value:
        return value
    path = Path(value).expanduser()
    looks_like_path = path.suffix.lower() == ".onnx" or "/" in value or "\\" in value
    if looks_like_path and not path.is_absolute():
        path = repo_dir / path
    return str(path) if looks_like_path else value

def load_config() -> AppConfig:
    repo_dir = Path(__file__).resolve().parents[1]  # repo root
    load_dotenv(repo_dir / ".env")

    wakeword = (os.getenv("WAKEWORD") or "").strip()
    thresh = float(os.getenv("THRESH", "0.6"))
    cooldown_s = float(os.getenv("COOLDOWN", "2.5"))

    command_wakewords_raw = (os.getenv("COMMAND_WAKEWORDS") or "").strip()
    command_wakewords = tuple(
        _normalize_wakeword_ref(item, repo_dir)
        for item in command_wakewords_raw.split(",")
        if item.strip()
    )
    command_thresh = float(os.getenv("COMMAND_THRESH", "0.75"))
    command_cooldown_s = float(os.getenv("COMMAND_COOLDOWN", "2.0"))

    ha_url = (os.getenv("HA_URL") or "").strip() or None
    ha_token = (os.getenv("HA_TOKEN") or "").strip() or None
    ha_language = (os.getenv("HA_LANGUAGE") or "en").strip()

    hub_url = (os.getenv("HUB_URL") or "").strip() or None
    hub_api_key = (os.getenv("HUB_API_KEY") or "").strip() or None
    hub_timeout_s = float(os.getenv("HUB_TIMEOUT", "5"))

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
        command_wakewords=command_wakewords,
        command_thresh=command_thresh,
        command_cooldown_s=command_cooldown_s,
        ha_url=ha_url,
        ha_token=ha_token,
        ha_language=ha_language,
        hub_url=hub_url,
        hub_api_key=hub_api_key,
        hub_timeout_s=hub_timeout_s,
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
