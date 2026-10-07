import asyncio
from collections.abc import Awaitable
from weakref import WeakKeyDictionary
from typing import Any

import httpx

from .openai_compat import OpenAICompatBackend

_runtime_locks: WeakKeyDictionary = WeakKeyDictionary()


def _runtime_lock(base_url: str) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    locks = _runtime_locks.setdefault(loop, {})
    return locks.setdefault(base_url, asyncio.Lock())


class OllamaBackend(OpenAICompatBackend):
    supports_single_model = True

    def __init__(
        self, base_url: str, timeout_s: float = 60, single_model_only: bool = True
    ):
        super().__init__(base_url, timeout_s)
        self.single_model_only = single_model_only
        openai_base_url = base_url.rstrip("/")
        self.native_base_url = (
            openai_base_url[:-3]
            if openai_base_url.endswith("/v1")
            else openai_base_url
        )
        self.native_client = httpx.AsyncClient(
            base_url=self.native_base_url, timeout=timeout_s
        )

    async def _unload(self, model: str) -> None:
        response = await self.native_client.post(
            "/api/generate",
            json={"model": model, "prompt": "", "stream": False, "keep_alive": 0},
        )
        response.raise_for_status()

    async def _unload_loaded_models(self) -> None:
        response = await self.native_client.get("/api/ps")
        response.raise_for_status()
        for loaded_model in response.json()["models"]:
            await self._unload(loaded_model["name"])

    async def _run_single_model(
        self, model: str, request: Awaitable[dict[str, Any]]
    ) -> dict[str, Any]:
        async with _runtime_lock(self.native_base_url):
            await self._unload_loaded_models()
            try:
                return await request
            finally:
                await self._unload(model)

    async def chat(self, payload: dict) -> dict:
        if not self.single_model_only:
            return await super().chat(payload)
        return await self._run_single_model(
            payload["model"], OpenAICompatBackend.chat(self, payload)
        )

    async def embeddings(self, payload: dict) -> dict:
        if not self.single_model_only:
            return await super().embeddings(payload)
        return await self._run_single_model(
            payload["model"], OpenAICompatBackend.embeddings(self, payload)
        )

    async def close(self) -> None:
        await super().close()
        await self.native_client.aclose()
