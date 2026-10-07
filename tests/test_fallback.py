import httpx
import pytest

from local_llm.router.fallback import call_with_fallback


class BackendStub:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.payloads = []

    async def chat(self, payload):
        self.payloads.append(payload)
        if self.error is not None:
            raise self.error
        return self.result


class RegistryStub:
    def __init__(self, chain, backends):
        self._chain = chain
        self.backends = backends

    def chain(self, name):
        assert name == "planner"
        return self._chain


def http_error(status_code):
    request = httpx.Request("POST", "http://localhost/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(
        f"HTTP {status_code}", request=request, response=response
    )


@pytest.mark.asyncio
async def test_falls_back_when_primary_returns_503():
    primary = BackendStub(error=http_error(503))
    backup = BackendStub(result={"choices": [{"message": {"content": "backup"}}]})
    registry = RegistryStub(
        [
            {"backend": "primary", "model": "model-a"},
            {"backend": "backup", "model": "model-b"},
        ],
        {"primary": primary, "backup": backup},
    )

    result = await call_with_fallback(
        registry, "planner", "chat", {"model": "local-llm", "messages": []}
    )

    assert result["choices"][0]["message"]["content"] == "backup"
    assert primary.payloads == [{"model": "model-a", "messages": []}]
    assert backup.payloads == [{"model": "model-b", "messages": []}]


@pytest.mark.asyncio
async def test_reports_failure_when_all_backends_fail():
    primary = BackendStub(error=http_error(503))
    backup = BackendStub(error=OSError("offline"))
    registry = RegistryStub(
        [
            {"backend": "primary", "model": "model-a"},
            {"backend": "backup", "model": "model-b"},
        ],
        {"primary": primary, "backup": backup},
    )

    with pytest.raises(RuntimeError, match="all backends failed for 'planner'"):
        await call_with_fallback(registry, "planner", "chat", {})
