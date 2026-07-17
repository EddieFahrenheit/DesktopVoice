from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class PlaywrightBridge:
    def __init__(
        self,
        *,
        command: str = "npx",
        args: tuple[str, ...] = ("@playwright/mcp@latest",),
        env: dict[str, str] | None = None,
    ) -> None:
        self._command = command
        self._args = args
        self._env = env
        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        if self._session is not None:
            return
        params = StdioServerParameters(
            command=self._command,
            args=list(self._args),
            env=self._env,
        )
        self._exit_stack = AsyncExitStack()
        read, write = await self._exit_stack.enter_async_context(stdio_client(params))
        self._session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()

    async def close(self) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
        self._session = None
        self._exit_stack = None

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("PlaywrightBridge not connected.")
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
