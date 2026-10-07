from pathlib import Path
from typing import Any

from .router.fallback import call_with_fallback
from .router.registry import Registry
from .toon.middleware import decode_assistant_content, encode_structured_messages


class LocalLLM:
    """Async Python client that routes directly to configured local model servers."""

    def __init__(self, config_path: str | Path, *, profile: str | None = None):
        self._registry = Registry(config_path)
        self.profile = profile
        if profile is not None:
            self._registry.chain(profile)

    def _resolve_model(self, model: str) -> str:
        if self.profile is not None:
            if model != "local-llm":
                raise ValueError(
                    "A profile-bound LocalLLM client must be called with model='local-llm'"
                )
            return self.profile
        if model == "local-llm":
            raise ValueError(
                "model='local-llm' requires a profile, e.g. LocalLLM(config_path, profile='planner')"
            )
        return model

    async def chat(
        self, *, model: str, messages: list[dict[str, Any]], **options: Any
    ) -> dict[str, Any]:
        logical_model = self._resolve_model(model)
        if self._registry.setting(logical_model, "toon_input"):
            messages = encode_structured_messages(messages)
        payload = {**options, "model": model, "messages": messages}
        response = await call_with_fallback(
            self._registry, logical_model, "chat", payload
        )
        if self._registry.setting(logical_model, "toon_output"):
            response = decode_assistant_content(response)
        return response

    async def embeddings(
        self, *, model: str, input: str | list[str], **options: Any
    ) -> dict[str, Any]:
        logical_model = self._resolve_model(model)
        payload = {**options, "model": model, "input": input}
        return await call_with_fallback(
            self._registry, logical_model, "embeddings", payload
        )

    async def health(self) -> dict[str, bool]:
        return {
            name: await backend.healthy()
            for name, backend in self._registry.backends.items()
        }

    async def close(self) -> None:
        for backend in self._registry.backends.values():
            await backend.close()

    async def __aenter__(self) -> "LocalLLM":
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.close()
