from __future__ import annotations

import pytest

from coder.config import Config
from coder.errors import ProviderError
from coder.lifecycle import Manager
from coder.models import load_registry
from coder.providers.base import (
    Billing, CreatedInstance, Instance, InstanceSpec, InstanceState, Offer, OfferQuery, Provider, SSHInfo,
)
from coder.session import SessionStore


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CODER_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("VAST_API_KEY", "test-key")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("coder.config.keychain_get", lambda name: None)
    return tmp_path / "home"


def make_offer(id=1, gpu="H100 SXM", n=2, ram=80, price=3.0, rel=0.99, geo="Texas, US", down=2000, disk=500, cpu=256) -> Offer:
    return Offer("fake", id, gpu, n, ram, cpu, disk, price, rel, geo, down)


class FakeProvider(Provider):
    name = "fake"

    def __init__(self, offers=None):
        self.offers = offers if offers is not None else [make_offer()]
        self.instances: dict[int, Instance] = {}
        self.destroyed: list[int] = []
        self.next_id = 100
        self.fail_destroy = 0
        self.unavailable_ids: set[int] = set()
        self.boot_state = InstanceState.RUNNING
        self.attached: list[int] = []

    def search_offers(self, query: OfferQuery):
        return list(self.offers)

    def create_instance(self, offer, spec: InstanceSpec):
        from coder.errors import OfferUnavailable
        if offer.id in self.unavailable_ids:
            raise OfferUnavailable("gone")
        self.next_id += 1
        self.instances[self.next_id] = Instance(
            "fake", self.next_id, self.boot_state, label=spec.label, ssh=SSHInfo("1.2.3.4", 2222),
            dph_total=offer.dph_total, gpu_name=offer.gpu_name, num_gpus=offer.num_gpus,
        )
        return CreatedInstance(self.next_id, "inst-key")

    def get_instance(self, instance_id):
        return self.instances.get(instance_id)

    def list_managed_instances(self):
        return [i for i in self.instances.values() if i.label.startswith("coder-managed")]

    def start_instance(self, instance_id): ...
    def stop_instance(self, instance_id): ...

    def destroy_instance(self, instance_id):
        if self.fail_destroy > 0:
            self.fail_destroy -= 1
            raise ProviderError("boom")
        self.destroyed.append(instance_id)
        self.instances.pop(instance_id, None)

    def attach_ssh_key(self, instance_id, public_key):
        self.attached.append(instance_id)

    def get_billing(self):
        return Billing(12.34)


class FakeUI:
    def __init__(self, answer=True):
        self.answer, self.messages, self.warnings, self.plans = answer, [], [], []

    def say(self, m): self.messages.append(m)
    def warn(self, m): self.warnings.append(m)
    def confirm(self, m, *, default=False): return self.answer
    def show_plan(self, *a): self.plans.append(a)


class FakeRemote:
    def __init__(self, fail_at: str | None = None):
        self.fail_at, self.calls = fail_at, []

    def _step(self, name):
        self.calls.append(name)
        if self.fail_at == name:
            from coder.errors import ProvisioningError
            raise ProvisioningError(f"{name} failed")

    def provision(self, *a, **k): self._step("provision")
    def start_watchdog(self, *a, **k): self._step("watchdog")
    def download_model(self, *a, **k): self._step("download")
    def start_server(self, *a, **k): self._step("server")
    def wait_healthy(self, *a, **k): self._step("healthy")


class FakeSSH:
    def wait_ready(self, *a, **k): ...


class TestManager(Manager):
    __test__ = False

    def __init__(self, *a, fail_at=None, **k):
        super().__init__(*a, **k)
        self.remote = FakeRemote(fail_at)
        self.tunnel_open: int | None = None
        self.busy: int | None = 0
        self.closed: list[int | None] = []

    def make_ssh(self, s): return FakeSSH()
    def make_remote(self, ssh): return self.remote
    def open_tunnel(self, s):
        self.tunnel_open = 4242
        return 4242
    def close_tunnel(self, pid):
        self.closed.append(pid)
        return True
    def tunnel_alive(self, s): return self.tunnel_open is not None and s.tunnel_pid == self.tunnel_open
    def server_ready(self, s): return True
    def active_requests(self, s): return self.busy


@pytest.fixture
def cfg(isolated_home) -> Config:
    c = Config()
    c.opencode.config_path = isolated_home / "oc" / "opencode.json"
    c.ssh.key_path = isolated_home / "ssh" / "id_test"
    return c


@pytest.fixture(autouse=True)
def fake_keygen(monkeypatch):
    monkeypatch.setattr("coder.lifecycle.generate_keypair", lambda p: "ssh-ed25519 AAAA test")


@pytest.fixture
def make_manager(cfg):
    def _make(provider=None, ui=None, **kw) -> TestManager:
        return TestManager(cfg, provider or FakeProvider(), load_registry(), SessionStore(), ui or FakeUI(), sleep=lambda s: None, **kw)
    return _make
