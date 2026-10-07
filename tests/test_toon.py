import json

import pytest

from local_llm.toon import middleware
from local_llm.toon.middleware import decode_assistant_content, encode_structured_messages


def test_non_json_untouched():
    msgs = [{"role": "user", "content": "hello"}]
    assert encode_structured_messages(msgs) == msgs


def test_decode_assistant_content_returns_json_text(monkeypatch):
    monkeypatch.setattr(middleware, "from_toon", lambda _text: {"ok": True})
    original = {
        "id": "chatcmpl-test",
        "choices": [{"message": {"role": "assistant", "content": "ok: true"}, "finish_reason": "stop"}],
    }

    decoded = decode_assistant_content(original)

    assert json.loads(decoded["choices"][0]["message"]["content"]) == {"ok": True}
    assert decoded["id"] == original["id"]
    assert original["choices"][0]["message"]["content"] == "ok: true"


def test_decode_assistant_content_preserves_non_text_choices():
    response = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": []}}]}
    assert decode_assistant_content(response) == response


def test_decode_assistant_content_propagates_invalid_toon(monkeypatch):
    def fail(_text):
        raise ValueError("invalid TOON")

    monkeypatch.setattr(middleware, "from_toon", fail)
    with pytest.raises(ValueError, match="invalid TOON"):
        decode_assistant_content({"choices": [{"message": {"content": "bad"}}]})
