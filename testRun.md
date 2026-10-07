Next, check the model backend:

Open http://127.0.0.1:8000/health. It reports whether the configured Ollama and llama.cpp servers are reachable.
Open http://127.0.0.1:8000/docs to try the API interactively. You can also see the configured logical model names at http://127.0.0.1:8000/v1/models.

- To use it from another Python project, activate that project’s virtual environment, then install the clone into it from this repo’s root:
  python -m pip install -e ".[toon]"
