import pytest
from typer.testing import CliRunner

from coder import cli
from coder.config import Config
from conftest import FakeProvider, TestManager
from coder.models import load_registry
from coder.session import SessionStore

runner = CliRunner()


@pytest.fixture
def patched(monkeypatch, cfg, make_manager):
    prov = FakeProvider()
    monkeypatch.setattr(cli, "_load", lambda: cfg)
    monkeypatch.setattr(cli, "build_provider", lambda c: prov)
    monkeypatch.setattr(cli, "_manager", lambda c: make_manager(prov, ui=cli.RichUI()))
    return prov


def test_models_lists_all_registered():
    r = runner.invoke(cli.app, ["models"])
    assert r.exit_code == 0
    for key in ("qwen3-coder-next", "qwen3-coder-30b-a3b", "devstral-small-2", "qwen3.8-27b", "gpt-oss-120b"):
        assert key in r.output
    assert "openai/gpt-oss-120b" in runner.invoke(cli.app, ["models", "--details"]).output


def test_status_without_session(patched):
    r = runner.invoke(cli.app, ["status"])
    assert r.exit_code == 0 and "No active session" in r.output


def test_offers_command(patched):
    r = runner.invoke(cli.app, ["offers", "--model", "qwen3-coder-next"])
    assert r.exit_code == 0 and "H100" in r.output


def test_start_status_stop_flow(patched):
    r = runner.invoke(cli.app, ["start", "--model", "qwen3-coder-next", "--yes"])
    assert r.exit_code == 0, r.output
    assert "Ready" in r.output and "127.0.0.1:18000/v1" in r.output
    assert len(patched.instances) == 1
    r = runner.invoke(cli.app, ["status"])
    assert "Instance" in r.output and "H100" in r.output
    r = runner.invoke(cli.app, ["stop", "--yes"])
    assert r.exit_code == 0 and "destroyed" in r.output and "Session cost" in r.output
    assert patched.instances == {}


def test_stop_asks_for_confirmation(patched):
    runner.invoke(cli.app, ["start", "--yes"])
    r = runner.invoke(cli.app, ["stop"], input="n\n")
    assert r.exit_code == 1 and len(patched.instances) == 1


def test_errors_are_clean(patched):
    r = runner.invoke(cli.app, ["start", "--model", "nope", "--yes"])
    assert r.exit_code == 1 and "Unknown model" in r.output and "Traceback" not in r.output


def test_cleanup_lists_and_destroys_only_tagged(patched):
    from coder.providers.base import Instance, InstanceState
    patched.instances[7] = Instance("fake", 7, InstanceState.RUNNING, label="coder-managed:old")
    patched.instances[8] = Instance("fake", 8, InstanceState.RUNNING, label="mine")
    r = runner.invoke(cli.app, ["cleanup"])
    assert "7" in r.output and 8 in patched.instances
    runner.invoke(cli.app, ["cleanup", "--destroy", "--yes"])
    assert 7 not in patched.instances and 8 in patched.instances


def test_config_show(monkeypatch, cfg):
    monkeypatch.setattr(cli, "load_config", lambda *a: Config())
    r = runner.invoke(cli.app, ["config"])
    assert r.exit_code == 0 and "max_hourly_price" in r.output
