import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException

from ..router.fallback import call_with_fallback
from ..router.registry import Registry
from ..toon.middleware import decode_assistant_content, encode_structured_messages

default_config = Path(__file__).resolve().parents[3] / "config" / "models.yaml"
reg = Registry(os.getenv("LOCAL_LLM_CONFIG", str(default_config)))
app = FastAPI(title="local-llm")


@app.get("/")
async def root():
    return {
        "name": "local-llm",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "models": "/v1/models",
        "chat_completions": "/v1/chat/completions",
        "embeddings": "/v1/embeddings",
    }


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": n, "object": "model"} for n in reg.models]}


@app.get("/health")
async def health():
    return {n: await b.healthy() for n, b in reg.backends.items()}


async def _dispatch(method: str, payload: dict):
    try:
        return await call_with_fallback(reg, payload.get("model", ""), method, payload)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))


@app.post("/v1/chat/completions")
async def chat(payload: dict):
    model = payload.get("model", "")
    try:
        reg.chain(model)
    except KeyError as e:
        raise HTTPException(404, str(e))
    if reg.setting(model, "toon_input"):
        payload["messages"] = encode_structured_messages(payload.get("messages", []))
    response = await _dispatch("chat", payload)
    if reg.setting(model, "toon_output"):
        try:
            response = decode_assistant_content(response)
        except (ValueError, TypeError) as e:
            raise HTTPException(502, f"Model returned invalid TOON output: {e}") from e
    return response


@app.post("/v1/embeddings")
async def embeddings(payload: dict):
    return await _dispatch("embeddings", payload)


def main():
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
