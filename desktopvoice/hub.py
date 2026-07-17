from contextlib import asynccontextmanager
import os
import shlex

from fastapi import FastAPI

from desktopvoice.config import load_config
from desktopvoice.ha_bridge import HomeAssistantBridge
from desktopvoice.playwright_bridge import PlaywrightBridge
from desktopvoice.hub_routes import router as hub_router


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_args(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return tuple(shlex.split(raw))


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()

    ha = HomeAssistantBridge.from_config(cfg)
    await ha.connect()
    app.state.ha = ha

    pw = None
    if _env_bool("PLAYWRIGHT_MCP_ENABLED", False):
        pw_command = os.getenv("PLAYWRIGHT_MCP_COMMAND", "npx")
        pw_args = _env_args(
            "PLAYWRIGHT_MCP_ARGS",
            ("@playwright/mcp@latest", "--cdp-endpoint", "http://127.0.0.1:9222"),
        )
        pw = PlaywrightBridge(command=pw_command, args=pw_args)
        await pw.connect()

    app.state.pw = pw

    try:
        yield
    finally:
        if pw is not None:
            await pw.close()
        await ha.close()


app = FastAPI(lifespan=lifespan)
app.include_router(hub_router, prefix="/hub")
