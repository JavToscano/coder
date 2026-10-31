import json

import pytest

from coder import costs, opencode
from coder.errors import CoderError, NoOffersError, ProvisioningError
from coder.providers.base import Instance, InstanceState
from coder.session import SessionStore
from conftest import FakeProvider, FakeUI, make_offer


def test_start_happy_path(make_manager, cfg):
    prov = FakeProvider()
    m = make_manager(prov)
    s, reused = m.start("qwen3-coder-next", yes=True)
    assert not reused and s.phase == "ready" and s.tunnel_pid == 4242
    assert s.label.startswith("coder-managed:") and prov.instances[s.instance_id].label == s.label
    assert prov.attached == [s.instance_id]
    assert m.remote.calls == ["provision", "watchdog", "download", "server", "healthy"]
    assert SessionStore().load().instance_id == s.instance_id
    d = json.loads(cfg.effective_opencode_path.read_text())
    assert d["model"] == "coder/Qwen3-Coder-Next" and d["provider"]["coder"]["options"]["apiKey"] == s.api_key
    assert s.deadline_epoch > s.created_at


def test_start_reuses_existing_session_and_reopens_dead_tunnel(make_manager):
    prov = FakeProvider()
    m = make_manager(prov)
    s, _ = m.start(yes=True)
    m.tunnel_open = None  # tunnel died (Mac slept)
    s2, reused = m.start(yes=True)
    assert reused and s2.instance_id == s.instance_id and len(prov.instances) == 1 and m.tunnel_open == 4242


@pytest.mark.parametrize("step", ["provision", "download", "server", "healthy"])
def test_failed_provisioning_destroys_instance_and_cleans_state(make_manager, cfg, step):
    prov = FakeProvider()
    m = make_manager(prov, fail_at=step)
    with pytest.raises(ProvisioningError):
        m.start(yes=True)
    assert prov.instances == {} and len(prov.destroyed) == 1
    assert SessionStore().load() is None
    assert not cfg.effective_opencode_path.exists()  # nothing left configured


def test_keyboard_interrupt_during_provisioning_still_cleans_up(make_manager):
    prov = FakeProvider()
    m = make_manager(prov)

    def boom(*a, **k):
        raise KeyboardInterrupt
    m.remote.download_model = boom
    with pytest.raises(KeyboardInterrupt):
        m.start(yes=True)
    assert prov.instances == {} and SessionStore().load() is None


def test_cleanup_failure_is_loud_and_keeps_session_for_retry(make_manager):
    prov = FakeProvider()
    prov.fail_destroy = 99
    ui = FakeUI()
    m = make_manager(prov, ui=ui, fail_at="download")
    with pytest.raises(ProvisioningError):
        m.start(yes=True)
    s = SessionStore().load()
    assert s.phase == "destroy_failed" and len(prov.instances) == 1
    assert any("COULD NOT CONFIRM DESTRUCTION" in w for w in ui.warnings)
    prov.fail_destroy = 0
    make_manager(prov).destroy(yes=True)
    assert prov.instances == {} and SessionStore().load() is None


def test_start_refuses_when_previous_session_is_broken(make_manager):
    prov = FakeProvider(); prov.fail_destroy = 99
    with pytest.raises(ProvisioningError):
        make_manager(prov, fail_at="download").start(yes=True)
    with pytest.raises(CoderError, match="coder destroy"):
        make_manager(prov).start(yes=True)


def test_boot_failure_state_triggers_cleanup(make_manager):
    prov = FakeProvider(); prov.boot_state = InstanceState.FAILED
    with pytest.raises(ProvisioningError, match="entered state"):
        make_manager(prov).start(yes=True)
    assert prov.instances == {}


def test_startup_timeout(make_manager, cfg):
    prov = FakeProvider(); prov.boot_state = InstanceState.PENDING
    cfg.limits.startup_timeout_minutes = 0.0001
    with pytest.raises(ProvisioningError, match="timeout"):
        make_manager(prov).start(yes=True)
    assert prov.instances == {}


def test_offer_gone_falls_back_to_next(make_manager):
    prov = FakeProvider([make_offer(1, price=2.0), make_offer(2, price=2.5)])
    prov.unavailable_ids = {1}
    s, _ = make_manager(prov).start(yes=True)
    assert prov.instances[s.instance_id].dph_total == 2.5


def test_no_offers_and_cost_cap(make_manager, cfg):
    with pytest.raises(NoOffersError):
        make_manager(FakeProvider([])).start(yes=True)
    cfg.limits.max_session_cost = 1.0
    prov = FakeProvider()
    with pytest.raises(CoderError, match="max_session_cost"):
        make_manager(prov).start(yes=True)
    assert prov.instances == {}  # refused before renting anything


def test_declining_confirmation_creates_nothing(make_manager):
    prov = FakeProvider()
    with pytest.raises(CoderError, match="Aborted"):
        make_manager(prov, ui=FakeUI(answer=False)).start()
    assert prov.instances == {}


def test_deadline_respects_cost_and_duration_caps(make_manager, cfg):
    cfg.limits.max_session_cost = 12.0  # at $3/h -> 4h cap, below the 6h duration cap
    s, _ = make_manager(FakeProvider()).start(yes=True)
    assert abs((s.deadline_epoch - s.created_at) / 3600 - 4.0) < 1e-6


def test_stop_destroys_restores_opencode_and_records_cost(make_manager, cfg):
    prov = FakeProvider()
    m = make_manager(prov)
    cfg.effective_opencode_path.parent.mkdir(parents=True)
    cfg.effective_opencode_path.write_text(json.dumps({"theme": "x"}))
    s, _ = m.start(yes=True)
    cost = m.stop(yes=True)
    assert cost is not None and prov.destroyed == [s.instance_id] and prov.instances == {}
    assert m.closed == [4242] and SessionStore().load() is None
    assert json.loads(cfg.effective_opencode_path.read_text()) == {"theme": "x"}
    assert costs.month_to_date()[1] == 1 and not opencode.is_applied()


def test_stop_warns_about_active_requests_and_can_abort(make_manager):
    prov = FakeProvider()
    ui = FakeUI()
    m = make_manager(prov, ui=ui)
    m.start(yes=True)
    m.busy = 2
    ui.answer = False
    with pytest.raises(CoderError, match="Aborted"):
        m.stop()
    assert any("2 inference request" in w for w in ui.warnings) and len(prov.instances) == 1
    ui.answer = True
    m.stop()
    assert prov.instances == {}


def test_stop_requires_confirmation_without_yes(make_manager):
    prov = FakeProvider()
    ui = FakeUI()
    m = make_manager(prov, ui=ui)
    m.start(yes=True)
    ui.answer = False
    with pytest.raises(CoderError):
        m.stop()
    assert len(prov.instances) == 1


def test_stop_without_session_is_a_noop(make_manager):
    prov = FakeProvider()
    assert make_manager(prov).stop(yes=True) is None and prov.destroyed == []


def test_destroy_verifies_and_retries(make_manager):
    prov = FakeProvider()
    m = make_manager(prov)
    m.start(yes=True)
    prov.fail_destroy = 2  # transient API failures
    m.destroy(yes=True)
    assert prov.instances == {} and SessionStore().load() is None


def test_status_warnings_and_reconcile_never_touch_foreign(make_manager, cfg):
    prov = FakeProvider()
    m = make_manager(prov)
    s, _ = m.start(yes=True)
    s.created_at -= 3600 * 3.5  # 3.5h at $3/h = $10.5 > $10 limit
    SessionStore().save(s)
    info = m.status()
    assert info.server_ready and info.tunnel_alive and any("limit" in w for w in info.warnings)

    prov.instances[900] = Instance("fake", 900, InstanceState.RUNNING, label="coder-managed:orphan")
    prov.instances[901] = Instance("fake", 901, InstanceState.RUNNING, label="someone-elses-job")
    orphans = m.find_orphans()
    assert [o.id for o in orphans] == [900]  # not the active session, not the foreign instance
    assert m.destroy_orphans(orphans + [prov.instances[901]]) == [900]  # even if handed a foreign one
    assert 901 in prov.instances and s.instance_id in prov.instances


def test_stale_session_cleared_when_instance_vanished(make_manager):
    prov = FakeProvider()
    m = make_manager(prov)
    s, _ = m.start(yes=True)
    prov.instances.clear()
    with pytest.raises(CoderError, match="Stale"):
        m.start(yes=True)
    assert SessionStore().load() is None
