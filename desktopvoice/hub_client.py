import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def send_hub_command(*, hub_url: str, text: str, api_key: str | None, timeout_s: float):
    url = f"{hub_url.rstrip('/')}/hub/command"
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
    except URLError as exc:
        return False, "network", str(exc)
