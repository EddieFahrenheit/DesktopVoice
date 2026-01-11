from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from desktopvoice.config import load_config
from desktopvoice.ha_bridge import HomeAssistantBridge
from desktopvoice.hub_routes import router as hub_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    bridge = HomeAssistantBridge.from_config(cfg)
    await bridge.connect()
    app.state.ha = bridge
    try:
        yield
    finally:
        await bridge.close()

app = FastAPI(lifespan=lifespan)
app.include_router(hub_router, prefix="/hub")