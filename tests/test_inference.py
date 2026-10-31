import pytest

from coder import inference
from coder.errors import ConfigError
from coder.models import load_registry
from coder.remote import Remote


def cmd(model, engine, tp, ctx=131072, variant=None):
    spec = load_registry().get(model)
    vn, v = spec.variant(variant)
    return inference.build_serve_command(spec, vn, v, engine, tp, ctx)


def test_vllm_command_binds_localhost_with_auth_and_tools():
    c = cmd("qwen3-coder-next", "vllm", 2)
    a = c.argv
    assert a[:2] == ["vllm", "serve"] and a[a.index("--host") + 1] == "127.0.0.1"
    assert a[a.index("--tensor-parallel-size") + 1] == "2"
    assert "--enable-auto-tool-choice" in a and a[a.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert c.api_key_flag == ["--api-key", '"$CODER_API_KEY"']  # expanded remotely; no secret in our argv
    assert c.shell().endswith('--api-key "$CODER_API_KEY"')


def test_sglang_command_and_reasoning_parser():
    c = cmd("qwen3.8-27b", "sglang", 1, variant="bf16")
    assert "sglang.launch_server" in c.argv and "--tp-size" in c.argv and "--context-length" in c.argv
    assert c.argv[c.argv.index("--reasoning-parser") + 1] == "qwen3"


def test_engine_and_gpu_validation():
    _, v = load_registry().get("qwen3-coder-next").variant(None)
    assert inference.pick_engine(v, "auto") == "vllm"
    with pytest.raises(ConfigError):
        inference.pick_engine(v, "llamacpp")
    with pytest.raises(ConfigError):
        inference.pick_gpu_count(v, 1)
    assert inference.pick_gpu_count(v, 4) == 4


def test_serve_script_sources_secrets_and_runs_offline():
    s = inference.serve_script(cmd("gpt-oss-120b", "vllm", 1))
    assert "secrets.env" in s and "HF_HUB_OFFLINE=1" in s and "exec vllm serve" in s


METRICS = '''# HELP vllm:num_requests_running x
vllm:num_requests_running{engine="0",model_name="m"} 2.0
vllm:num_requests_waiting{engine="0",model_name="m"} 1.0
'''


def test_metrics_parsing():
    assert inference.parse_active_requests(METRICS) == 3
    assert inference.parse_active_requests("sglang:num_running_reqs{a=\"b\"} 0\nsglang:num_queue_reqs{a=\"b\"} 0\n") == 0
    assert inference.parse_active_requests("nothing here") is None


def test_models_ready_checks_auth_and_name(monkeypatch):
    import httpx

    def fake_get(url, headers, timeout):
        ok = headers["Authorization"] == "Bearer tok"
        return httpx.Response(200 if ok else 401, json={"data": [{"id": "M"}]})

    monkeypatch.setattr(inference.httpx, "get", fake_get)
    assert inference.models_ready("http://x/v1", "tok", "M")
    assert not inference.models_ready("http://x/v1", "bad", "M")
    assert not inference.models_ready("http://x/v1", "tok", "other")


def test_download_script_variants():
    direct = Remote._download_script("a/b", "/root/models/x", "", None, False)
    assert "snapshot_download" in direct and "aws" not in direct and "--delete" not in direct
    from coder.config import StorageConfig
    st = StorageConfig(enabled=True, bucket="bk", prefix="p", endpoint_url="https://s3.example")
    s3 = Remote._download_script("a/b", "/root/models/x", "s3://bk/p/a/b", st, True)
    assert "s3 sync" in s3 and "--delete" not in s3 and "--endpoint-url" in s3
    assert "integrity check failed" in s3 and "cache hit" in s3 and "cache miss" in s3
