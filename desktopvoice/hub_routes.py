from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from .ha_bridge import HomeAssistantBridge

router = APIRouter()


class CommandPayload(BaseModel):
    text: str


class CommandResult(BaseModel):
    ok: bool
    action: str | None = None


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


OFF_KEYWORDS = ("off", "shutdown", "power down", "turn off", "sleep", "close", "kill", "boss", "fuck")
ON_KEYWORDS = ("on", "wake", "start", "turn on", "power on", "open", "wait")

INTENTS = [
    {
        "aliases": {"jarvis", "jarv", "gervais", "server", "amrit", "i'm right", "i'm ripped"},
        "type": "script",
        "on": "script.turn_on_jarvis",
        "off": "script.turn_off_jarvis",
    },
    {
        "aliases": {"work", "workstation", "desk"},
        "type": "script",
        "on": "script.wake_workstation",
        "off": "script.sleep_workstation",
    },
    {
        "aliases": {"mac", "rhasspy"},
        "type": "script",
        "on": "script.full_wake_rhasspy",
        "off": None,
    },
    {
        "aliases": {"bed", "bedroom", "bedroom lights", "bet", "then", "that"},
        "type": "entity",
        "entity": "light.bedroom_lights",
    },
    {
        "aliases": {"main", "living room", "living room lights", "me", "make", "lay", "man", "may", "wait", "made", "lane"},
        "type": "entity",
        "entity": "light.living_room_lights",
    },
    {
        "aliases": {"downstairs", "downstairs lights"},
        "type": "entity",
        "entity": "light.downstairs_light",
    },
]

ALIAS_INDEX = []
for entry in INTENTS:
    for alias in entry["aliases"]:
        ALIAS_INDEX.append((alias, entry))

ALIAS_INDEX.sort(key=lambda x: len(x[0]), reverse=True)


async def _dispatch_to_ha(ha: HomeAssistantBridge, text: str) -> str:
    t = _normalize(text)

    for alias, entry in ALIAS_INDEX:
        if alias in t:
            entry_type = entry.get("type")
            if entry_type == "script":
                if any(k in t for k in OFF_KEYWORDS):
                    off_entity = entry.get("off")
                    if not off_entity:
                        raise HTTPException(status_code=422, detail="No off script defined for this device.")
                    await ha.call_service(domain="script", service="turn_on", service_data={"entity_id": off_entity})
                    return f"script.turn_on:{off_entity}"

                on_entity = entry.get("on")
                if not on_entity:
                    raise HTTPException(status_code=422, detail="No on script defined for this device.")
                await ha.call_service(domain="script", service="turn_on", service_data={"entity_id": on_entity})
                return f"script.turn_on:{on_entity}"

            if entry_type == "entity":
                entity_id = entry.get("entity")
                if not entity_id:
                    raise HTTPException(status_code=422, detail="Missing entity id for device.")
                domain = entity_id.split(".", 1)[0]
                if any(k in t for k in OFF_KEYWORDS):
                    await ha.call_service(domain=domain, service="turn_off", service_data={"entity_id": entity_id})
                    return f"{domain}.turn_off:{entity_id}"
                if any(k in t for k in ON_KEYWORDS):
                    await ha.call_service(domain=domain, service="turn_on", service_data={"entity_id": entity_id})
                    return f"{domain}.turn_on:{entity_id}"
                raise HTTPException(status_code=422, detail="Missing on/off intent for device.")

            raise HTTPException(status_code=422, detail="Unsupported intent type.")

    raise HTTPException(status_code=422, detail="No matching device found.")


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/command", response_model=CommandResult)
async def command(payload: CommandPayload, request: Request) -> CommandResult:
    ha: HomeAssistantBridge = request.app.state.ha
    action = await _dispatch_to_ha(ha, payload.text)
    return CommandResult(ok=True, action=action)
