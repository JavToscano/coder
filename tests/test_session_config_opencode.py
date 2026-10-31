import json

import pytest

from coder import opencode, paths
from coder.config import Config, load_config
from coder.errors import ConfigError, OpenCodeConfigError
from coder.session import Session, SessionStore


def _session(**kw):
    base = dict(provider="fake", instance_id=1, label="coder-managed:x", model_key="m", variant="v", engine="vllm",
                served_name="M", repo="a/b", context_length=1000, dph=2.0)
    return Session(**{**base, **kw})


def test_session_roundtrip_private_and_cost(isolated_home):
    st = SessionStore()
    s = _session(created_at=1000.0, api_key="secret")
    st.save(s)
    assert (st.path.stat().st_mode & 0o777) == 0o600
    assert st.load().api_key == "secret"
    assert abs(s.cost(1000.0 + 5400) - 3.0) < 1e-9
    st.clear()
    assert st.load() is None


def test_session_lock_is_exclusive(isolated_home):
    from coder.errors import CoderError
    st = SessionStore()
    with st.lock():
        with pytest.raises(CoderError, match="already running"):
            with SessionStore().lock():
                pass


def test_config_defaults_and_validation(isolated_home, tmp_path):
    assert load_config(tmp_path / "missing.yaml").provider == "vast"
    f = tmp_path / "c.yaml"
    f.write_text("limits:\n  max_hourly_price: -1\nbogus: 1\n")
    with pytest.raises(ConfigError) as e:
        load_config(f)
    assert "limits.max_hourly_price" in str(e.value) and "bogus" in str(e.value)
    f.write_text("model: qwen3-coder-30b-a3b\nidle:\n  enabled: true\n  timeout_minutes: 20\n")
    c = load_config(f)
    assert c.model == "qwen3-coder-30b-a3b" and c.idle.enabled and c.idle.timeout_minutes == 20


def test_config_has_no_secret_fields():
    assert not any("key" in k and k != "key_path" for k in Config.model_fields)


BLOCK = dict(display_name="Q", model_id="Q", base_url="http://127.0.0.1:18000/v1", api_key="tok", context=1000, output=250, tool_call=True)


def _apply(path, set_default=True):
    return opencode.apply(path, "coder", opencode.provider_block(**BLOCK), "Q", set_default_model=set_default)


def test_opencode_creates_valid_provider_and_removes_file_on_restore(isolated_home):
    p = isolated_home / "oc" / "opencode.json"
    assert _apply(p) == "coder/Q"
    d = json.loads(p.read_text())
    prov = d["provider"]["coder"]
    assert prov["npm"] == "@ai-sdk/openai-compatible" and prov["options"]["baseURL"].endswith("/v1")
    assert prov["models"]["Q"]["limit"] == {"context": 1000, "output": 250} and prov["models"]["Q"]["tool_call"] is True
    assert d["model"] == "coder/Q" and d["$schema"] == opencode.SCHEMA_URL
    assert (p.stat().st_mode & 0o777) == 0o600
    assert opencode.restore() and not p.exists()
    assert not opencode.restore()  # nothing left


def test_opencode_preserves_user_settings_and_restores_exactly(isolated_home):
    p = isolated_home / "oc" / "opencode.json"
    p.parent.mkdir(parents=True)
    original = {"$schema": "https://opencode.ai/config.json", "theme": "tokyonight", "model": "anthropic/claude-x",
                "provider": {"other": {"npm": "x", "models": {}}, "coder": {"npm": "old"}}}
    p.write_text(json.dumps(original))
    _apply(p)
    _apply(p)  # idempotent: second apply must not clobber the true originals in the backup
    d = json.loads(p.read_text())
    assert d["theme"] == "tokyonight" and "other" in d["provider"] and d["provider"]["coder"]["npm"] == "@ai-sdk/openai-compatible"
    assert opencode.restore()
    assert json.loads(p.read_text()) == original


def test_opencode_restore_keeps_model_if_user_switched(isolated_home):
    p = isolated_home / "oc" / "opencode.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"model": "anthropic/a"}))
    _apply(p)
    d = json.loads(p.read_text()); d["model"] = "openai/b"; p.write_text(json.dumps(d))
    opencode.restore()
    assert json.loads(p.read_text()) == {"model": "openai/b"}


def test_opencode_refuses_invalid_or_jsonc(isolated_home):
    p = isolated_home / "oc" / "opencode.json"
    p.parent.mkdir(parents=True)
    p.write_text("{ // comment\n}")
    with pytest.raises(OpenCodeConfigError, match="not valid JSON"):
        _apply(p)
    p.unlink()
    p.with_suffix(".jsonc").write_text("{}")
    with pytest.raises(OpenCodeConfigError, match="JSONC"):
        _apply(p)


def test_opencode_without_default_model_leaves_model(isolated_home):
    p = isolated_home / "oc" / "opencode.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"model": "a/b"}))
    _apply(p, set_default=False)
    assert json.loads(p.read_text())["model"] == "a/b"


def test_dotenv_parsing_and_precedence(isolated_home, tmp_path, monkeypatch):
    from coder.config import load_dotenv, parse_dotenv
    assert parse_dotenv("# c\nexport A=1\nB='x y'\nC=\"q\" \nD=v # note\nbad line\n") == {"A": "1", "B": "x y", "C": "q", "D": "v"}

    proj = tmp_path / "proj"; proj.mkdir()
    (proj / ".env").write_text("VAST_API_KEY=from-cwd\nHF_TOKEN=hf-cwd\nPATH=/evil\nEXTRA=1\n")
    cfgdir = isolated_home / "config"; cfgdir.mkdir(parents=True)
    (cfgdir / ".env").write_text("VAST_API_KEY=from-config\nAWS_ACCESS_KEY_ID=ak\n")
    monkeypatch.chdir(proj)
    for k in ("VAST_API_KEY", "HF_TOKEN", "AWS_ACCESS_KEY_ID", "EXTRA"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HF_TOKEN", "from-shell")
    import os
    path_before = os.environ["PATH"]
    assert len(load_dotenv()) == 2
    assert os.environ["VAST_API_KEY"] == "from-cwd"      # cwd beats config dir
    assert os.environ["HF_TOKEN"] == "from-shell"        # real env beats .env
    assert os.environ["AWS_ACCESS_KEY_ID"] == "ak"
    assert os.environ["PATH"] == path_before and "EXTRA" not in os.environ  # allow-list only
    for k in ("VAST_API_KEY", "AWS_ACCESS_KEY_ID"):
        monkeypatch.delenv(k)
