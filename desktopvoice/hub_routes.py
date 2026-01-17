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


ENTITY_KEYWORDS: list[tuple[str, str]] = [
    ("bedroom lights", "light.bedroom_lights"),
    ("living room lights", "light.living_room_lights"),
    ("downstairs lights", "light.downstairs_light"),
]

SCRIPT_ACTIONS: list[tuple[str, str, str | None]] = [
    ("jarvis", "script.turn_on_jarvis", "script.turn_off_jarvis"),
    ("workstation", "script.wake_workstation", "script.sleep_workstation"),
    ("rhasspy", "script.full_wake_rhasspy", None),
]


OFF_KEYWORDS = ("off", "shutdown", "power down", "turn off", "sleep")
ON_KEYWORDS = ("on", "wake", "start", "turn on", "power on")


async def _dispatch_to_ha(ha: HomeAssistantBridge, text: str) -> str:
    t = _normalize(text)

    for keyword, on_entity, off_entity in SCRIPT_ACTIONS:
        if keyword in t:
            if any(k in t for k in OFF_KEYWORDS):
                if not off_entity:
                    raise HTTPException(status_code=422, detail="No off script defined for this device.")
                await ha.call_service(domain="script", service="turn_on", service_data={"entity_id": off_entity})
                return f"script.turn_on:{off_entity}"

            await ha.call_service(domain="script", service="turn_on", service_data={"entity_id": on_entity})
            return f"script.turn_on:{on_entity}"

    for keyword, entity_id in ENTITY_KEYWORDS:
        if keyword in t:
            domain = entity_id.split(".", 1)[0]
            if any(k in t for k in OFF_KEYWORDS):
                await ha.call_service(domain=domain, service="turn_off", service_data={"entity_id": entity_id})
                return f"{domain}.turn_off:{entity_id}"
            if any(k in t for k in ON_KEYWORDS):
                await ha.call_service(domain=domain, service="turn_on", service_data={"entity_id": entity_id})
                return f"{domain}.turn_on:{entity_id}"
            raise HTTPException(status_code=422, detail="Missing on/off intent for device.")

    raise HTTPException(status_code=422, detail="No matching device found.")


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/command", response_model=CommandResult)
async def command(payload: CommandPayload, request: Request) -> CommandResult:
    ha: HomeAssistantBridge = request.app.state.ha
    action = await _dispatch_to_ha(ha, payload.text)
    return CommandResult(ok=True, action=action)
