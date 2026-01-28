import json
import socket
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def send_hub_action(*, hub_url: str, action: str, api_key: str | None, timeout_s: float):
    url = f"{hub_url.rstrip('/')}/hub/action"
    data = json.dumps({"action": action}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    req = Request(url, data=data, headers=headers, method="POST")

    try:
        with urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
            return True, resp.status, body
    except HTTPError as exc:
        return False, exc.code, exc.read().decode("utf-8")
    except (TimeoutError, socket.timeout) as exc:
        return False, "timeout", str(exc)
    except URLError as exc:
        return False, "network", str(exc)

def send_hub_llm(*, hub_url: str, text: str, api_key: str | None, timeout_s: float):
    url = f"{hub_url.rstrip('/')}/hub/llm"
    data = json.dumps({"text": text}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    req = Request(url, data=data, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
            return True, resp.status, body
    except HTTPError as exc:
        return False, exc.code, exc.read().decode("utf-8")
    except (TimeoutError, socket.timeout) as exc:
        return False, "timeout", str(exc)
    except URLError as exc:
        return False, "network", str(exc)
