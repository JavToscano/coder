from types import SimpleNamespace

import pytest

from coder import remote as remote_mod
from coder.errors import ProvisioningError
from coder.remote import Remote, last_log_line


def test_last_log_line_takes_latest_progress_frame():
    text = "INFO start\nLoading shards:  0%\rLoading shards: 50%\rLoading shards: 100%\n\n"
    assert last_log_line(text) == "Loading shards: 100%"
    assert last_log_line("\x1b[32mINFO\x1b[0m ready\n") == "INFO ready"
    assert last_log_line("\n  \n") == ""


class FakeSSH:
    def __init__(self, local_ok=False):
        self.local_ok, self.cmds = local_ok, []

    def run(self, cmd, **kw):
        self.cmds.append(cmd)
        if "pgrep" in cmd:
            return SimpleNamespace(stdout="up")
        if "curl" in cmd:
            return SimpleNamespace(stdout="200" if self.local_ok else "000")
        return SimpleNamespace(stdout="INFO Capturing CUDA graphs 50%\n")


def _remote(ssh, clock):
    msgs = []
    r = Remote(ssh, log=msgs.append, sleep=lambda s: clock.advance(s))
    return r, msgs


class Clock:
    t = 0.0
    def __call__(self): return self.t
    def advance(self, s): self.t += s


def test_wait_healthy_reports_progress_and_times_out(monkeypatch):
    monkeypatch.setattr(remote_mod, "models_ready", lambda *a, **k: False)
    clock = Clock()
    r, msgs = _remote(FakeSSH(), clock)
    with pytest.raises(ProvisioningError):
        r.wait_healthy("http://x/v1", "k", "m", 100, progress_every_s=30, clock=clock)
    assert any("Capturing CUDA graphs" in m for m in msgs)
    assert not any("tunnel" in m for m in msgs)


def test_wait_healthy_flags_broken_tunnel(monkeypatch):
    monkeypatch.setattr(remote_mod, "models_ready", lambda *a, **k: False)
    clock = Clock()
    r, msgs = _remote(FakeSSH(local_ok=True), clock)
    with pytest.raises(ProvisioningError):
        r.wait_healthy("http://x/v1", "k", "m", 100, progress_every_s=30, clock=clock)
    assert sum("tunnel" in m for m in msgs) == 1
