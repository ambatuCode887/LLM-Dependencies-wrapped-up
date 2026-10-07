# local-llm

Importable Python library and optional OpenAI-compatible HTTP gateway for local models,
with named model chains (fallback) and optional TOON encoding.

## Use from another Python project

Activate the consuming project's virtual environment, then install this clone into
that environment. Run the install command from this repository's root:

```powershell
python -m pip install -e ".[toon]"
```

`-e` installs this local checkout for development. Ollama (or another configured
backend) and the model files remain separate prerequisites; they are not downloaded
by this package.

Configure one profile per agent in `config/models.yaml`. Each profile can have its own
ordered model chain and TOON overrides. Give each agent a client bound to its profile;
all agents can use the same stable model identifier, `local-llm`:

```python
import asyncio
from pathlib import Path

from local_llm import LocalLLM

CONFIG = Path(r"C:\path\to\local-llm\config\models.yaml")

async def main():
    async with (
        LocalLLM(CONFIG, profile="planner") as planner,
        LocalLLM(CONFIG, profile="coder") as coder,
    ):
        plan = await planner.chat(
            model="local-llm",
            messages=[{"role": "user", "content": "Plan a small RAG pipeline."}],
        )
        code = await coder.chat(
            model="local-llm",
            messages=[{"role": "user", "content": "Write a Python function to split text into chunks."}],
        )
        print(plan["choices"][0]["message"]["content"])
        print(code["choices"][0]["message"]["content"])


asyncio.run(main())
```

The `profile` selects the configured chain while `model="local-llm"` stays the same in
each agent's call. `LocalLLM` sends requests directly to the configured Ollama or
llama.cpp OpenAI-compatible servers. Use `profile="embedder"` with
`await embedder.embeddings(model="local-llm", input="text")` for embeddings.
`await llm.health()` checks configured backends, and `async with` closes HTTP connections.
Pass backend-supported chat options as keyword arguments, for example `temperature=0.2`.
Without a profile, the existing named model calls such as `model="planner"` continue
to work. Each profile can override `toon_input` and `toon_output`; otherwise the
top-level settings apply.

## Offline operation

After Python dependencies, the backend runtime, and model files are installed locally,
requests go only to configured backend URLs (the sample config uses localhost). No
cloud model or internet connection is needed at inference time. Initial package
installation and downloading model files may require internet access; model updates
are a separate maintenance task.

With `single_model_only: true`, Ollama requests from this Python process are
serialized. Before each request, the adapter unloads models currently loaded in that
Ollama runtime; after the request, it unloads the selected model. This saves memory
but adds model-switch latency and unloads models used by other apps on that Ollama
instance. Route all agents through one Python process to share the lock. This strict
mode currently requires Ollama for every configured model chain; other backend types
must implement their own unload adapter before they can be used with it.

The HTTP gateway described below is optional; use it when a non-Python application or
multiple independent applications need a shared API endpoint.

## Multi-agent RAG example

After the backend and configured models are running, from the repository root:

```powershell
python examples\multi_agent_rag.py
```

The example embeds a few local passages with the `embedder` profile, retrieves the
closest passages in memory, then calls separate `planner` and `coder` profiles. Each
agent uses `model="local-llm"`; its bound profile selects the configured local model
chain. It uses no cloud API or external vector database.

## Install and run on Windows

From the repository folder in PowerShell:

```powershell
python -m pip install -e ".[toon,dev]"
python -m uvicorn local_llm.gateway.app:app --reload
```

This starts the gateway at `http://127.0.0.1:8000`. Using `python -m uvicorn` avoids
the Windows PATH warning for the installed `local-llm` script.

In another PowerShell window, check it:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/v1/models
```

Start Ollama or llama.cpp separately and edit `config/models.yaml` to use model names
available on your backend. If your config is elsewhere, set `LOCAL_LLM_CONFIG` to its
absolute path before starting the server.

Point any project at it: `base_url="http://127.0.0.1:8000/v1"`, `model="planner"`.

Edit `config/models.yaml` to define logical models and their fallback chains.
Backends: Ollama and llama.cpp server (both OpenAI-compatible).

Layout: `gateway/` API, `backends/` adapters, `router/` registry + fallback, `toon/` codec + middleware.

### Optional TOON input and output

Set `toon_input: true` to encode JSON objects/arrays in user messages as TOON. Set
`toon_output: true` for profiles whose prompts require valid TOON output. The gateway
decodes assistant content back to compact JSON text while preserving the
OpenAI-compatible response shape, so a caller can parse `message.content` as JSON.
Invalid TOON output produces an HTTP 502 error rather than being passed through silently.

### Token benchmark

Install the optional benchmark dependencies and run the sample:

```powershell
python -m pip install -e ".[toon,benchmark]"
python examples\toon_benchmark.py
```

It compares a uniform sample of retrieved documents using `cl100k_base` via tiktoken.
This is an estimate only; exact counts depend on the tokenizer used by your local model.
