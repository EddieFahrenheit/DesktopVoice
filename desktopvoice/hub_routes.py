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

# desktopvoice/hub_routes.py
class ActionPayload(BaseModel):
    action: str

ACTION_MAP = {
    "main_on": {"domain": "light", "service": "turn_on", "data": {"entity_id": "light.living_room_lights"}},
    "kill_main": {"domain": "light", "service": "turn_off", "data": {"entity_id": "light.living_room_lights"}},
    "bed_on": {"domain": "light", "service": "turn_on", "data": {"entity_id": "light.bedroom_lights"}},
    "kill_bed": {"domain": "light", "service": "turn_off", "data": {"entity_id": "light.bedroom_lights"}},
    "down_stairs_on": {"domain": "light", "service": "turn_on", "data": {"entity_id": "light.downstairs_light"}},
    "kill_down_stairs": {"domain": "light", "service": "turn_off", "data": {"entity_id": "light.downstairs_light"}},
    "wake_work_stay_shin": {"domain": "script", "service": "turn_on", "data": {"entity_id": "script.wake_workstation"}},
    "wake_ahm_riht": {"domain": "script", "service": "turn_on", "data": {"entity_id": "script.turn_on_jarvis"}},
}

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

@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
