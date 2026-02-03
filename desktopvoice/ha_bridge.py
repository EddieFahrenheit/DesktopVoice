from __future__ import annotations

import asyncio
import logging
import os
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import AppConfig, load_config

logger = logging.getLogger(__name__)

DEFAULT_MCP_COMMAND = "uvx"
DEFAULT_MCP_ARGS = ("ha-mcp",)
MCP_CALL_SERVICE_TOOL = "ha_call_service"


class HomeAssistantBridge:
    def __init__(
        self,
        *,
        ha_url: str,
        ha_token: str,
        ha_language: str = "en",
        mcp_command: str = DEFAULT_MCP_COMMAND,
        mcp_args: tuple[str, ...] = DEFAULT_MCP_ARGS,
    ) -> None:
        if not ha_url:
            raise ValueError("ha_url is required.")
        if not ha_token:
            raise ValueError("ha_token is required.")
        self._ha_url = ha_url
        self._ha_token = ha_token
        self._ha_language = ha_language or "en"
        self._mcp_command = mcp_command
        self._mcp_args = mcp_args
        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._lock = asyncio.Lock()

    @classmethod
    def from_config(cls, cfg: AppConfig) -> "HomeAssistantBridge":
        if not cfg.ha_url or not cfg.ha_token:
            raise RuntimeError("Set HA_URL and HA_TOKEN in .env before using HomeAssistantBridge.")
        return cls(ha_url=cfg.ha_url, ha_token=cfg.ha_token, ha_language=cfg.ha_language)

    async def __aenter__(self) -> "HomeAssistantBridge":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def connect(self) -> None:
        if self._session is not None:
            return

        env = os.environ.copy()
        env["HOMEASSISTANT_URL"] = self._ha_url
        env["HOMEASSISTANT_TOKEN"] = self._ha_token
        if self._ha_language:
            env["HOMEASSISTANT_LANGUAGE"] = self._ha_language

        server_params = StdioServerParameters(
            command=self._mcp_command,
            args=list(self._mcp_args),
            env=env,
        )

        self._exit_stack = AsyncExitStack()
        read, write = await self._exit_stack.enter_async_context(stdio_client(server_params))
        self._session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        logger.info("Home Assistant MCP session initialized.")

    async def close(self) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
        self._exit_stack = None
        self._session = None

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("HomeAssistantBridge not connected. Call connect() first.")
        return self._session

    async def list_tools(self) -> list[str]:
        session = self._require_session()
        async with self._lock:
            result = await session.list_tools()
        return [tool.name for tool in result.tools]

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        session = self._require_session()
        async with self._lock:
            result = await session.call_tool(name, arguments=arguments or {})
        if hasattr(result, "model_dump"):
            return result.model_dump()
        if hasattr(result, "dict"):
            return result.dict()
        return result

    async def call_service(
        self,
        *,
        domain: str,
        service: str,
        service_data: dict[str, Any] | None = None,
    ) -> Any:
        session = self._require_session()
        payload = {"domain": domain, "service": service, "data": service_data or {}}
        async with self._lock:
            return await session.call_tool(MCP_CALL_SERVICE_TOOL, arguments=payload)


    async def switch_turn_on(self, entity_id: str) -> None:
        await self.call_service(domain="switch", service="turn_on", service_data={"entity_id": entity_id})

    async def switch_turn_off(self, entity_id: str) -> None:
        await self.call_service(domain="switch", service="turn_off", service_data={"entity_id": entity_id})

    async def run_shell_command(self, command_name: str) -> None:
        await self.call_service(domain="shell_command", service=command_name, service_data={})


if __name__ == "__main__":
    async def main() -> None:
        cfg = load_config()
        bridge = HomeAssistantBridge.from_config(cfg)
        async with bridge:
            await bridge.switch_turn_on("switch.jarvis")
            await bridge.run_shell_command("pc_monitor_on")

    asyncio.run(main())
