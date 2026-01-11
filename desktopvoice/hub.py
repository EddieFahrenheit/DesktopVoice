from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from desktopvoice.config import load_config
from desktopvoice.ha_bridge import HomeAssistantBridge
