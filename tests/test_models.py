import pytest
import yaml

from coder.errors import ConfigError
from coder.models import load_registry


def test_builtin_models_present():
    reg = load_registry()
    for k in ("qwen3-coder-next", "qwen3-coder-30b-a3b", "devstral-small-2", "qwen3.8-27b", "gpt-oss-120b"):
        reg.get(k)


def test_qwen_next_not_single_gpu():
    _, v = load_registry().get("qwen3-coder-next").variant(None)
    assert 1 not in v.gpus and v.min_total_vram_gb > 80
    _, bf16 = load_registry().get("qwen3-coder-next").variant("bf16")
    assert bf16.min_total_vram_gb > 160


def test_unknown_model_and_variant():
    reg = load_registry()
    with pytest.raises(ConfigError, match="Unknown model"):
        reg.get("nope")
    with pytest.raises(ConfigError, match="variant"):
        reg.get("gpt-oss-120b").variant("fp8")


def test_context_limit():
    m = load_registry().get("gpt-oss-120b")
    assert m.effective_context(None) == 131072
    with pytest.raises(ConfigError):
        m.effective_context(10**7)


def test_user_registry_extends_and_validates(isolated_home, tmp_path):
    f = tmp_path / "models.yaml"
    f.write_text(yaml.safe_dump({"models": {"mine": {
        "display_name": "Mine", "served_name": "mine", "context_length": 8192, "default_variant": "a",
        "variants": {"a": {"repo": "x/y", "quant": "awq", "download_gb": 5, "min_total_vram_gb": 8,
                           "min_gpu_vram_gb": 8, "gpus": [1], "engines": ["vllm"], "tool_call_parser": "hermes"}}}}}))
    assert "mine" in load_registry(f).keys()
    f.write_text(yaml.safe_dump({"models": {"bad": {"display_name": "x"}}}))
    with pytest.raises(ConfigError, match="bad"):
        load_registry(f)
