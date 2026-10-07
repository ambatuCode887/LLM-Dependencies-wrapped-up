import pytest

from local_llm.router.registry import Registry


def test_single_model_mode_rejects_backend_without_unload_support(tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text(
        """\
backends:
  compatible:
    type: openai_compatible
    base_url: http://localhost:8080/v1
models:
  planner:
    chain:
      - {backend: compatible, model: local}
settings:
  single_model_only: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot enforce single-model unloading"):
        Registry(config)


@pytest.mark.asyncio
async def test_profile_settings_override_global_defaults(tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text(
        """\
backends:
  ollama:
    type: ollama
    base_url: http://localhost:11434/v1
models:
  planner:
    chain:
      - {backend: ollama, model: model-a}
    toon_output: false
settings:
  single_model_only: true
  toon_output: true
""",
        encoding="utf-8",
    )
    registry = Registry(config)

    assert registry.setting("planner", "toon_output") is False
    assert registry.setting("planner", "toon_input") is False
    await registry.backends["ollama"].close()
