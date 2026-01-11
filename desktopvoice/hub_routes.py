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
    ("jarvis", "switch.jarvis"),
    ("server computer", "switch.server_computer"),
    ("server", "switch.server_computer"),
    ("ubuntu pc", "switch.ubuntu_pc"),
    ("coding mac", "switch.macbook_pro"),
    ("macbook pro", "switch.macbook_pro"),
    ("kitchen mac", "switch.rhasspy"),
    ("rhasspy", "switch.rhasspy"),
]

OFF_KEYWORDS = ("off", "shutdown", "power down", "turn off", "sleep")
ON_KEYWORDS = ("on", "wake", "start", "turn on", "power on")


async def _dispatch_to_ha(ha: HomeAssistantBridge, text: str) -> str:
    t = _normalize(text)

    if "monitor" in t and any(k in t for k in ON_KEYWORDS):
        await ha.run_shell_command("pc_monitor_on")
        return "shell_command.pc_monitor_on"

    if "monitor" in t and any(k in t for k in OFF_KEYWORDS):
        await ha.run_shell_command("pc_monitor_off")
        return "shell_command.pc_monitor_off"

    for keyword, entity_id in ENTITY_KEYWORDS:
        if keyword in t:
            if any(k in t for k in OFF_KEYWORDS):
                await ha.switch_turn_off(entity_id)
                return f"switch.turn_off:{entity_id}"
            if any(k in t for k in ON_KEYWORDS):
                await ha.switch_turn_on(entity_id)
                return f"switch.turn_on:{entity_id}"
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
