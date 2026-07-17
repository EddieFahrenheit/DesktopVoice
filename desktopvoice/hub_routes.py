from __future__ import annotations

import json
import os
import asyncio
import time

from typing import Any
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest, urlopen

from .ha_bridge import HomeAssistantBridge

router = APIRouter()
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "glm-4.7-flash:q4"
DEFAULT_OLLAMA_TIMEOUT_S = 30.0

DEFAULT_SYSTEM_PROMPT = (
    "You control Home Assistant and browser tools via MCP. "
    "Return ONLY JSON: {\"tool\":\"ha:<tool_name>|pw:<tool_name>\",\"arguments\":{...}}. "
    "Use ha:... for Home Assistant and pw:... for Playwright. "
    "If no tool fits, return {\"tool\":\"none\",\"response\":\"...\"}."
)

ALLOWED_DOMAINS = {"light", "switch", "script"}
ALLOWED_SERVICES = {"turn_on", "turn_off", "toggle"}

DISCOVERY_DOMAINS = {
    d.strip()
    for d in (os.getenv("ENTITY_DISCOVERY_DOMAINS", "")).split(",")
    if d.strip()
}
if not DISCOVERY_DOMAINS:
    DISCOVERY_DOMAINS = set(ALLOWED_DOMAINS)

ENTITY_CACHE_TTL_S = float(os.getenv("ENTITY_CACHE_TTL", "300"))
ENTITY_PROMPT_LIMIT = int(os.getenv("ENTITY_PROMPT_LIMIT", "200"))

async def _namespaced_tool_map(request: Request) -> dict[str, tuple[str, str]]:
    ha: HomeAssistantBridge = request.app.state.ha
    out = {f"ha:{name}": ("ha", name) for name in await ha.list_tools()}

    pw = getattr(request.app.state, "pw", None)
    if pw is not None:
        for name in await pw.list_tools():
            out[f"pw:{name}"] = ("pw", name)
    return out

def _normalize_entity(entry: dict[str, Any]) -> dict[str, str] | None:
    entity_id = entry.get("entity_id")
    if not isinstance(entity_id, str) or "." not in entity_id:
        return None
    domain = entry.get("domain") or entity_id.split(".", 1)[0]
    if domain not in DISCOVERY_DOMAINS:
        return None
    name = (
        entry.get("name")
        or entry.get("friendly_name")
        or entry.get("label")
        or entry.get("title")
        or ""
    )
    return {"entity_id": entity_id, "name": str(name), "domain": str(domain)}

def _collect_entities(obj: Any) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if isinstance(obj, dict):
        normalized = _normalize_entity(obj)
        if normalized:
            found.append(normalized)
        for value in obj.values():
            found.extend(_collect_entities(value))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_collect_entities(item))
    return found

def _dedupe_entities(entities: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    for entry in entities:
        entity_id = entry["entity_id"]
        if entity_id in seen:
            continue
        seen.add(entity_id)
        result.append(entry)
    return result

def _format_entity_context(entities: list[dict[str, str]], limit: int) -> str:
    lines = []
    for entry in entities[:limit]:
        name = entry["name"]
        if name:
            lines.append(f"- {entry['entity_id']} ({name})")
        else:
            lines.append(f"- {entry['entity_id']}")
    return "\n".join(lines)

async def _discover_entities(ha: HomeAssistantBridge) -> list[dict[str, str]]:
    all_entities: list[dict[str, str]] = []

    # Prefer per-domain search (empty query often returns nothing).
    for domain in sorted(DISCOVERY_DOMAINS):
        query = f"{domain}."
        try:
            result = await ha.call_tool("ha_search_entities", {"query": query, "limit": 200})
        except Exception:
            continue
        all_entities.extend(_extract_entities_from_result(result))

    if all_entities:
        return _dedupe_entities(all_entities)

    # Fallback: overview (often big, but includes entities)
    try:
        result = await ha.call_tool("ha_get_overview", {})
        entities = _extract_entities_from_result(result)
        if entities:
            return entities
    except Exception:
        pass

    return []

def _parse_json_text(text: str) -> list[dict | list]:
    text = text.strip()
    payloads: list[dict | list] = []
    try:
        parsed = json.loads(text)
        payloads.append(parsed)
        return payloads
    except json.JSONDecodeError:
        pass

    for open_char, close_char in (("{", "}"), ("[", "]")):
        start = text.find(open_char)
        end = text.rfind(close_char)
        if start != -1 and end != -1 and end > start:
            try:
                payloads.append(json.loads(text[start : end + 1]))
            except json.JSONDecodeError:
                pass
    return payloads


def _extract_payloads(result: Any) -> list[Any]:
    payloads: list[Any] = []
    if result is None:
        return payloads
    if isinstance(result, (dict, list)):
        payloads.append(result)

    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        payloads.extend(_parse_json_text(text))
    return payloads


def _extract_entities_from_result(result: Any) -> list[dict[str, str]]:
    entities: list[dict[str, str]] = []
    for payload in _extract_payloads(result):
        entities.extend(_collect_entities(payload))
    return _dedupe_entities(entities)


class EntityCache:
    def __init__(self, ttl_s: float) -> None:
        self._ttl_s = ttl_s
        self._last_refresh = 0.0
        self._entities: list[dict[str, str]] = []
        self._lock = asyncio.Lock()

    async def get(self, ha: HomeAssistantBridge) -> list[dict[str, str]]:
        now = time.time()
        if self._entities and (now - self._last_refresh) < self._ttl_s:
            return self._entities
        async with self._lock:
            now = time.time()
            if self._entities and (now - self._last_refresh) < self._ttl_s:
                return self._entities
            self._entities = await _discover_entities(ha)
            self._last_refresh = now
            return self._entities

    async def refresh(self, ha: HomeAssistantBridge) -> list[dict[str, str]]:
        async with self._lock:
            self._entities = await _discover_entities(ha)
            self._last_refresh = time.time()
            return self._entities

ENTITY_CACHE = EntityCache(ttl_s=ENTITY_CACHE_TTL_S)

def _entity_id_list(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    return []

def _ollama_chat(*, base_url: str, model: str, messages: list[dict], timeout_s: float) -> str:
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {"model": model, "messages": messages, "stream": False}
    data = json.dumps(payload).encode("utf-8")
    req = UrlRequest(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError) as exc:
        raise HTTPException(status_code=502, detail=f"Ollama error: {exc}") from exc
    return body["message"]["content"]

def _extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None

class CommandPayload(BaseModel):
    text: str


class CommandResult(BaseModel):
    ok: bool
    action: str | None = None

# desktopvoice/hub_routes.py
class ActionPayload(BaseModel):
    action: str

ACTION_MAP = {
    "main_on": {"domain": "light", "service": "turn_on", "data": {"entity_id": "light.living_room_lights"}},
    "kill_main": {"domain": "light", "service": "turn_off", "data": {"entity_id": "light.living_room_lights"}},
    "bed_on": {"domain": "light", "service": "turn_on", "data": {"entity_id": "light.bedroom_lights"}},
    "hey_luna": {"domain": "light", "service": "turn_on", "data": {"entity_id": "light.bedroom_lights"}},
    "kill_bed": {"domain": "light", "service": "turn_off", "data": {"entity_id": "light.bedroom_lights"}},
    "ok_home": {"domain": "light", "service": "turn_on", "data": {"entity_id": "light.downstairs_light"}},
    "kill_down_stairs": {"domain": "light", "service": "turn_off", "data": {"entity_id": "light.downstairs_light"}},
    "wake_work_stay_shin": {"domain": "script", "service": "turn_on", "data": {"entity_id": "script.wake_workstation"}},
    "wake_ahm_riht": {"domain": "script", "service": "turn_on", "data": {"entity_id": "script.turn_on_jarvis"}},
}

ALLOWED_ENTITY_IDS = {
    "light.living_room_lights",
    "light.bedroom_lights",
    "light.downstairs_light",
    "script.wake_workstation",
    "script.kill_workstation",
    "script.turn_on_jarvis",
    "script.turn_off_jarvis",
}

def _entity_id_list(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    return []

@router.post("/action", response_model=CommandResult)
async def action(payload: ActionPayload, request: Request) -> CommandResult:
    ha: HomeAssistantBridge = request.app.state.ha
    action = ACTION_MAP.get(payload.action)
    if not action:
        raise HTTPException(status_code=422, detail="Unknown action")
    await ha.call_service(
        domain=action["domain"],
        service=action["service"],
        service_data=action.get("data", {}),
    )
    return CommandResult(ok=True, action=payload.action)

@router.post("/llm", response_model=CommandResult)
async def llm_command(payload: CommandPayload, request: Request) -> CommandResult:
    ha: HomeAssistantBridge = request.app.state.ha

    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
    model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    timeout_s = float(os.getenv("OLLAMA_TIMEOUT", str(DEFAULT_OLLAMA_TIMEOUT_S)))
    system_prompt = os.getenv("OLLAMA_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

    entities = await ENTITY_CACHE.get(ha)
    if entities:
        entity_context = _format_entity_context(entities, ENTITY_PROMPT_LIMIT)
        if entity_context:
            system_prompt = f"{system_prompt}\n\nAvailable entities:\n{entity_context}"

    tool_map = await _namespaced_tool_map(request)
    available_tools = sorted(tool_map.keys())
    if available_tools:
        system_prompt = f"{system_prompt}\n\nAvailable tools:\n- " + "\n- ".join(available_tools)

    content = _ollama_chat(
        base_url=base_url,
        model=model,
        timeout_s=timeout_s,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": payload.text},
        ],
    )

    data = _extract_json(content)
    if not data:
        raise HTTPException(status_code=422, detail="LLM did not return valid JSON.")

    tool = data.get("tool")
    if tool == "none":
        return CommandResult(ok=True, action=None)
    if not isinstance(tool, str) or not tool.strip():
        raise HTTPException(status_code=422, detail="Missing or invalid tool.")

    args = data.get("arguments") or {}
    if not isinstance(args, dict):
        raise HTTPException(status_code=422, detail="arguments must be an object.")

    target = tool_map.get(tool)
    if target is None and ":" not in tool:
        target = tool_map.get(f"ha:{tool}")  # backward compatibility

    if target is None:
        raise HTTPException(status_code=422, detail=f"Unknown tool: {tool}")

    backend, raw_tool = target
    if backend == "ha":
        await ha.call_tool(raw_tool, args)
    else:
        pw = getattr(request.app.state, "pw", None)
        if pw is None:
            raise HTTPException(status_code=503, detail="Playwright MCP is disabled.")
        await pw.call_tool(raw_tool, args)

    return CommandResult(ok=True, action=tool)

@router.get("/tools")
async def all_tools(request: Request):
    tool_map = await _namespaced_tool_map(request)
    return {"tools": sorted(tool_map.keys())}


@router.get("/pw/tools")
async def pw_tools(request: Request):
    pw = getattr(request.app.state, "pw", None)
    if pw is None:
        return {"enabled": False, "tools": []}
    return {"enabled": True, "tools": await pw.list_tools()}

@router.get("/ha/tools")
async def ha_tools(request: Request):
    ha: HomeAssistantBridge = request.app.state.ha
    return {"tools": await ha.list_tools()}

@router.get("/entities")
async def get_entities(request: Request):
    ha: HomeAssistantBridge = request.app.state.ha
    return {"entities": await ENTITY_CACHE.get(ha)}

@router.post("/entities/refresh")
async def refresh_entities(request: Request):
    ha: HomeAssistantBridge = request.app.state.ha
    entities = await ENTITY_CACHE.refresh(ha)
    return {"count": len(entities)}

@router.get("/ha/debug/search")
async def debug_search(request: Request):
    ha: HomeAssistantBridge = request.app.state.ha
    return await ha.call_tool("ha_search_entities", {"query": "light.", "limit": 50})