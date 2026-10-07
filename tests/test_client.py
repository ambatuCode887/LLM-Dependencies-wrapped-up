import pytest

from local_llm import client as client_module
from local_llm.client import LocalLLM


class RegistryStub:
    def __init__(self, settings=None, models=None):
        self.settings = settings or {}
        self.models = models or {}
        self.backends = {}

    def setting(self, name, key, default=False):
        return self.models.get(name, {}).get(key, self.settings.get(key, default))


@pytest.mark.asyncio
async def test_chat_routes_logical_model_and_options(monkeypatch):
    llm = LocalLLM.__new__(LocalLLM)
    llm._registry = RegistryStub()
    llm.profile = None
    calls = []

    async def fake_call(registry, model, method, payload):
        calls.append((registry, model, method, payload))
        return {"choices": []}

    monkeypatch.setattr(client_module, "call_with_fallback", fake_call)
    messages = [{"role": "user", "content": "hello"}]

    result = await llm.chat(model="planner", messages=messages, temperature=0.2)

    assert result == {"choices": []}
    assert calls[0][1:] == (
        "planner",
        "chat",
        {"temperature": 0.2, "model": "planner", "messages": messages},
    )


@pytest.mark.asyncio
async def test_chat_applies_toon_settings(monkeypatch):
    llm = LocalLLM.__new__(LocalLLM)
    llm._registry = RegistryStub(settings={"toon_input": True, "toon_output": True})
    llm.profile = None
    calls = []

    async def fake_call(registry, model, method, payload):
        calls.append(payload)
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "users[1]{id}:\n  1"}}
            ]
        }

    monkeypatch.setattr(client_module, "call_with_fallback", fake_call)
    monkeypatch.setattr(
        client_module, "encode_structured_messages", lambda messages: [{"role": "user", "content": "TOON"}]
    )

    result = await llm.chat(
        model="planner",
        messages=[{"role": "user", "content": '[{"id":1}]'}],
    )

    assert calls[0]["messages"] == [{"role": "user", "content": "TOON"}]
    assert result["choices"][0]["message"]["content"] == '{"users":[{"id":1}]}'


@pytest.mark.asyncio
async def test_profile_client_routes_local_llm_to_profile(monkeypatch):
    llm = LocalLLM.__new__(LocalLLM)
    llm._registry = RegistryStub()
    llm.profile = "researcher"
    calls = []

    async def fake_call(registry, model, method, payload):
        calls.append((model, method, payload))
        return {"choices": []}

    monkeypatch.setattr(client_module, "call_with_fallback", fake_call)

    await llm.chat(
        model="local-llm",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert calls == [
        (
            "researcher",
            "chat",
            {
                "model": "local-llm",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    ]


@pytest.mark.asyncio
async def test_profile_client_rejects_different_model_name():
    llm = LocalLLM.__new__(LocalLLM)
    llm._registry = RegistryStub()
    llm.profile = "researcher"

    with pytest.raises(ValueError, match="must be called with model='local-llm'"):
        await llm.chat(model="planner", messages=[])


@pytest.mark.asyncio
async def test_local_llm_alias_requires_a_profile():
    llm = LocalLLM.__new__(LocalLLM)
    llm._registry = RegistryStub()
    llm.profile = None

    with pytest.raises(ValueError, match="requires a profile"):
        await llm.chat(model="local-llm", messages=[])


@pytest.mark.asyncio
async def test_profile_can_override_global_toon_output(monkeypatch):
    llm = LocalLLM.__new__(LocalLLM)
    llm._registry = RegistryStub(
        settings={"toon_input": False, "toon_output": True},
        models={"researcher": {"toon_output": False}},
    )
    llm.profile = "researcher"
    raw_response = {
        "choices": [{"message": {"role": "assistant", "content": "plain response"}}]
    }

    async def fake_call(registry, model, method, payload):
        return raw_response

    monkeypatch.setattr(client_module, "call_with_fallback", fake_call)

    result = await llm.chat(model="local-llm", messages=[])

    assert result == raw_response
