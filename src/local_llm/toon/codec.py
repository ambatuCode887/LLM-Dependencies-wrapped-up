"""TOON helpers with a compact-JSON fallback for installations without the optional extra."""
import json

try:
    from toon import encode as _encode, decode as _decode  # python-toon
except ImportError:
    _encode = _decode = None

TOON_AVAILABLE = _encode is not None


def to_toon(data) -> str:
    if _encode is None:
        return json.dumps(data, separators=(",", ":"))
    return _encode(data)


def from_toon(text: str):
    if _decode is None:
        return json.loads(text)
    return _decode(text)
