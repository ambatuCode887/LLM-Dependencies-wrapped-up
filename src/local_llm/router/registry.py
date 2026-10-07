from pathlib import Path

import yaml

from ..backends.openai_compat import OpenAICompatBackend
from ..backends.ollama import OllamaBackend


class Registry:
    def __init__(self, path: str | Path):
        cfg = yaml.safe_load(Path(path).read_text())
        self.settings = cfg.get("settings", {})
        timeout = self.settings.get("timeout_s", 60)
        single_model_only = self.settings.get("single_model_only", True)
        backend_configs = cfg["backends"]
        self.models = cfg["models"]
        for profile, profile_cfg in self.models.items():
            for entry in profile_cfg["chain"]:
                backend_name = entry["backend"]
                if backend_name not in backend_configs:
                    raise ValueError(
                        f"profile '{profile}' references unknown backend "
                        f"'{backend_name}'"
                    )
                backend_type = backend_configs[backend_name].get(
                    "type", "openai_compatible"
                )
                if single_model_only and backend_type != "ollama":
                    raise ValueError(
                        f"profile '{profile}' uses backend '{backend_name}', "
                        "which cannot enforce single-model unloading"
                    )

        self.backends: dict[str, OpenAICompatBackend] = {}
        for name, backend_cfg in backend_configs.items():
            backend_type = backend_cfg.get("type", "openai_compatible")
            if backend_type == "ollama":
                backend = OllamaBackend(
                    backend_cfg["base_url"],
                    timeout,
                    single_model_only=single_model_only,
                )
            elif backend_type == "openai_compatible":
                backend = OpenAICompatBackend(backend_cfg["base_url"], timeout)
            else:
                raise ValueError(
                    f"unsupported backend type '{backend_type}' for '{name}'"
                )
            self.backends[name] = backend

    def chain(self, name: str) -> list[dict[str, str]]:
        if name not in self.models:
            raise KeyError(f"unknown model '{name}'")
        return self.models[name]["chain"]

    def setting(self, name: str, key: str, default: bool = False) -> bool:
        if name not in self.models:
            raise KeyError(f"unknown model '{name}'")
        return self.models[name].get(key, self.settings.get(key, default))
