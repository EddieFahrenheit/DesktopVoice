import json
import os
import urllib.request

HA_URL = os.getenv("HA_URL", "").rstrip("/")
HA_TOKEN = os.getenv("HA_TOKEN", "")
HA_LANGUAGE = os.getenv("HA_LANGUAGE", "en")

def send_to_ha(text: str) -> dict:
    """
    Send transcribed text to Home Assistant's conversation API.
    Returns HA's JSON response (intent + speech).
    """
    if not HA_URL or not HA_TOKEN:
        raise RuntimeError("HA_URL/HA_TOKEN not set")

    payload = {
        "text": text,
        "language": HA_LANGUAGE,
    }

    req = urllib.request.Request(
        f"{HA_URL}/api/conversation/process",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {HA_TOKEN}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)
