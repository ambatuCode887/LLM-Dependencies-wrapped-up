"""Ollama and llama.cpp server both expose OpenAI-style endpoints, so one adapter covers both."""
import httpx


class OpenAICompatBackend:
    supports_single_model = False

    def __init__(self, base_url: str, timeout_s: float = 60):
        self.client = httpx.AsyncClient(base_url=base_url, timeout=timeout_s)

    async def chat(self, payload: dict) -> dict:
        r = await self.client.post("/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()

    async def embeddings(self, payload: dict) -> dict:
        r = await self.client.post("/embeddings", json=payload)
        r.raise_for_status()
        return r.json()

    async def healthy(self) -> bool:
        try:
            return (await self.client.get("/models", timeout=3)).status_code == 200
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        await self.client.aclose()
