import httpx
import pytest

from local_llm.backends.ollama import OllamaBackend
from local_llm.backends.ollama import _runtime_lock


class ResponseStub:
    def __init__(self, result):
        self.result = result

    def raise_for_status(self):
        return None

    def json(self):
        return self.result


class NativeClientStub:
    def __init__(self, loaded):
        self.loaded = list(loaded)
        self.calls = []

    async def get(self, path):
        self.calls.append(("GET", path))
        return ResponseStub(
            {"models": [{"name": model} for model in self.loaded]}
        )

    async def post(self, path, json):
        self.calls.append(("POST", path, json))
        if path == "/api/generate":
            if json["model"] in self.loaded:
                self.loaded.remove(json["model"])
        return ResponseStub({})


class OpenAIClientStub:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    async def post(self, path, json):
        self.calls.append((path, json))
        if self.error is not None:
            raise self.error
        return ResponseStub({"model": json["model"], "choices": []})


async def make_backend(loaded, error=None, single_model_only=True):
    backend = OllamaBackend(
        "http://localhost:11434/v1",
        single_model_only=single_model_only,
    )
    await backend.close()
    backend.client = OpenAIClientStub(error)
    backend.native_client = NativeClientStub(loaded)
    return backend


@pytest.mark.asyncio
async def test_single_model_request_unloads_existing_and_requested_models():
    backend = await make_backend(["old-model"])

    result = await backend.chat(
        {"model": "new-model", "messages": [{"role": "user", "content": "hi"}]}
    )

    assert result["model"] == "new-model"
    assert backend.native_client.loaded == []
    assert backend.native_client.calls == [
        ("GET", "/api/ps"),
        (
            "POST",
            "/api/generate",
            {
                "model": "old-model",
                "prompt": "",
                "stream": False,
                "keep_alive": 0,
            },
        ),
        (
            "POST",
            "/api/generate",
            {
                "model": "new-model",
                "prompt": "",
                "stream": False,
                "keep_alive": 0,
            },
        ),
    ]


@pytest.mark.asyncio
async def test_model_is_unloaded_when_request_fails():
    request = httpx.Request("POST", "http://localhost/v1/chat/completions")
    error = httpx.HTTPStatusError(
        "service unavailable",
        request=request,
        response=httpx.Response(503, request=request),
    )
    backend = await make_backend([], error=error)

    with pytest.raises(httpx.HTTPStatusError, match="service unavailable"):
        await backend.chat({"model": "model-a", "messages": []})

    assert backend.native_client.loaded == []
    assert backend.native_client.calls[-1][2]["keep_alive"] == 0


@pytest.mark.asyncio
async def test_unload_policy_can_be_disabled():
    backend = await make_backend([], single_model_only=False)

    await backend.chat({"model": "model-a", "messages": []})

    assert backend.native_client.calls == []


@pytest.mark.asyncio
async def test_clients_sharing_an_ollama_runtime_share_a_lock():
    assert _runtime_lock("http://localhost:11434") is _runtime_lock(
        "http://localhost:11434"
    )
