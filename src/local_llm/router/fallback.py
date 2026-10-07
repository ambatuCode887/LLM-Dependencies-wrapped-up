import logging

import httpx

from .registry import Registry

log = logging.getLogger("local_llm.router")


async def call_with_fallback(reg: Registry, name: str, method: str, payload: dict) -> dict:
    """Try each entry in the model's chain until one succeeds."""
    last_err: Exception | None = None
    for entry in reg.chain(name):
        backend = reg.backends[entry["backend"]]
        try:
            body = {**payload, "model": entry["model"]}
            return await getattr(backend, method)(body)
        except (httpx.HTTPError, OSError) as e:
            log.warning("%s/%s failed: %s", entry["backend"], entry["model"], e)
            last_err = e
    raise RuntimeError(f"all backends failed for '{name}'") from last_err
