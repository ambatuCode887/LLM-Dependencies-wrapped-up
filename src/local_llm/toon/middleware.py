import json

from .codec import from_toon, to_toon


def encode_structured_messages(messages: list[dict]) -> list[dict]:
    """Re-encode user messages whose content is a JSON object/array as TOON."""
    out = []
    for m in messages:
        c = m.get("content")
        if m.get("role") == "user" and isinstance(c, str) and c.lstrip()[:1] in "{[":
            try:
                m = {**m, "content": to_toon(json.loads(c))}
            except ValueError:
                pass
        out.append(m)
    return out


def decode_assistant_content(response: dict) -> dict:
    """Decode TOON assistant content to compact JSON text without changing the API shape."""
    choices = response.get("choices")
    if not isinstance(choices, list):
        return response

    decoded_choices = []
    for choice in choices:
        if not isinstance(choice, dict):
            decoded_choices.append(choice)
            continue
        message = choice.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            decoded_choices.append(choice)
            continue

        value = from_toon(message["content"])
        decoded_message = {**message, "content": json.dumps(value, separators=(",", ":"), ensure_ascii=False)}
        decoded_choices.append({**choice, "message": decoded_message})

    return {**response, "choices": decoded_choices}
