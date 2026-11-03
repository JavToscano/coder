"""Session orchestration: start (with guaranteed cleanup on failure), stop/destroy, reconcile."""
from __future__ import annotations

import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from . import costs, inference, offers as offers_mod, opencode
from .config import Config, get_secret
from .errors import (
    CoderError, NoOffersError, OfferUnavailable, ProviderError, ProvisioningError,
)
from .inference import REMOTE_PORT
from .models import ModelSpec, Registry, Variant
from .providers.base import (
    MANAGED_LABEL_PREFIX, Instance, InstanceSpec, InstanceState, Offer, Provider,
)
from .remote import Remote
from .session import Session, SessionStore
from .ssh import SSHClient, generate_keypair, pid_is_ssh, stop_tunnel


class UI(Protocol):
    def say(self, msg: str) -> None: ...
    def warn(self, msg: str) -> None: ...
    def confirm(self, msg: str, *, default: bool = False) -> bool: ...
    def show_plan(self, offer: Offer, est: costs.Estimate, model: ModelSpec, variant_name: str, variant: Variant, engine: str, context: int) -> None: ...


@dataclass
class StatusInfo:
    session: Session
    instance: Instance | None
    instance_error: str = ""
    server_ready: bool = False
    tunnel_alive: bool = False
    cost: float = 0.0
    month_cost: float = 0.0
    month_sessions: int = 0
    warnings: list[str] = field(default_factory=list)


class Manager:
    def __init__(
        self, cfg: Config, provider: Provider, registry: Registry, store: SessionStore, ui: UI,
        *, sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg, self.provider, self.registry, self.store, self.ui = cfg, provider, registry, store, ui
        self.sleep = sleep

    # Seams replaced in tests -------------------------------------------------
    def make_ssh(self, s: Session) -> SSHClient:
        return SSHClient(s.ssh_host, s.ssh_port, s.ssh_user, Path(s.ssh_key), Path(s.known_hosts),
                         connect_timeout=self.cfg.ssh.connect_timeout_s)

    def make_remote(self, ssh: SSHClient) -> Remote:
        return Remote(ssh, log=self.ui.say, sleep=self.sleep)

    def open_tunnel(self, s: Session) -> int:
        return self.make_ssh(s).start_tunnel(s.local_port, s.remote_port)

    def close_tunnel(self, pid: int | None) -> bool:
        return stop_tunnel(pid)

    def tunnel_alive(self, s: Session) -> bool:
        return pid_is_ssh(s.tunnel_pid)

    def server_ready(self, s: Session) -> bool:
        return inference.models_ready(s.base_url, s.api_key, s.served_name)

    def active_requests(self, s: Session) -> int | None:
        return inference.active_requests(s.local_port)

    # --------------------------------------------------------------------- start
    def start(self, model_key: str | None = None, *, variant: str | None = None, engine: str | None = None,
              context: int | None = None, yes: bool = False, offer_rank: int | None = None) -> tuple[Session, bool]:
        """Returns (session, reused)."""
        with self.store.lock():
            existing = self.store.load()
            if existing:
                return self._reuse(existing), True
            return self._start_new(model_key or self.cfg.model, variant, engine, context, yes, offer_rank), False

    def _reuse(self, s: Session) -> Session:
        if s.phase != "ready":
            raise CoderError(
                f"A previous session ({s.phase}, instance {s.instance_id}) did not finish cleanly. "
                "Run `coder destroy` to remove it before starting a new one."
            )
        inst = self.provider.get_instance(s.instance_id)
        if inst is None or inst.state in (InstanceState.GONE, InstanceState.FAILED, InstanceState.STOPPED):
            self.ui.warn(f"Session instance {s.instance_id} is {'gone' if inst is None else inst.state.value}; clearing stale session.")
            self._finish(s, destroyed_confirmed=inst is None)
            raise CoderError("Stale session cleared. Run `coder start` again.")
        if not self.tunnel_alive(s):
            self.ui.say("Re-establishing SSH tunnel…")
            s.tunnel_pid = self.open_tunnel(s)
            self.store.save(s)
        self._apply_opencode(s)
        return s

    def _start_new(self, model_key: str, variant_arg: str | None, engine_arg: str | None,
                   context_arg: int | None, yes: bool, offer_rank: int | None = None) -> Session:
        cfg = self.cfg
        spec = self.registry.get(model_key)
        vname, variant = spec.variant(variant_arg or cfg.variant)
        engine = inference.pick_engine(variant, engine_arg or cfg.engine)
        context = spec.effective_context(context_arg or cfg.context_length)
        hf_token = get_secret("HF_TOKEN", "hf", required=False)
        aws_env = self._aws_env() if cfg.storage.enabled else {}
        if cfg.storage.enabled and not cfg.storage.bucket:
            raise CoderError("storage.enabled is true but storage.bucket is empty")

        candidates = offers_mod.find_offers(self.provider, cfg, spec, variant)
        if not candidates:
            raise NoOffersError(
                f"No offers satisfy {spec.display_name}/{vname}: GPUs {variant.gpus}, >= {variant.min_gpu_vram_gb:g} GB/GPU, "
                f">= {variant.min_total_vram_gb:g} GB total, <= ${cfg.limits.max_hourly_price}/h, reliability >= {cfg.offers.min_reliability}. "
                "Raise limits.max_hourly_price, relax offers.*, or try a smaller model/variant."
            )
        if offer_rank is not None and not 1 <= offer_rank <= len(candidates):
            raise CoderError(f"--offer {offer_rank} is out of range: {len(candidates)} offers match (see `coder offers`).")
        # An explicit pick is never silently swapped for another host, so it gets no fallback list.
        candidates = candidates[offer_rank - 1:offer_rank] if offer_rank else candidates
        best = candidates[0]
        est = costs.estimate(best, variant.download_gb, cfg.limits.expected_session_hours)
        self.ui.show_plan(best, est, spec, vname, variant, engine, context)

        if est.session_cost > cfg.limits.max_session_cost:
            raise CoderError(
                f"Estimated session cost ${est.session_cost:.2f} exceeds limits.max_session_cost ${cfg.limits.max_session_cost:.2f}."
            )
        try:
            bill = self.provider.get_billing()
        except ProviderError as e:
            bill = None
            self.ui.warn(f"Could not read account balance ({e}); continuing.")
        need = est.startup_cost + best.dph_total  # startup plus at least one hour of use
        if bill and bill.balance_usd is not None and bill.balance_usd < need:
            raise CoderError(
                f"Account balance ${bill.balance_usd:.2f} is below the ~${need:.2f} needed to start "
                f"(startup + 1 h). Add credit at https://cloud.vast.ai/billing/ and retry."
            )
        spent, n = costs.month_to_date()
        if spent + est.session_cost > cfg.limits.monthly_budget:
            self.ui.warn(f"Month-to-date ${spent:.2f} ({n} sessions) + this session ${est.session_cost:.2f} exceeds your ${cfg.limits.monthly_budget:.0f} monthly budget.")
            if not yes and not self.ui.confirm("Continue anyway?"):
                raise CoderError("Aborted.")
        self.ui.warn("Code you work on is sent to the cloud host's GPU machine while the model is running.")
        if not yes and not self.ui.confirm(
            f"Rent {best.num_gpus}x {best.gpu_name} at ${best.dph_total:.2f}/h and start {spec.display_name}?", default=True
        ):
            raise CoderError("Aborted.")

        key_path = cfg.effective_ssh_key
        pubkey = generate_keypair(key_path)
        sid = secrets.token_hex(5)
        label = f"{MANAGED_LABEL_PREFIX}:{sid}"
        image = cfg.images[engine]
        disk = offers_mod.required_disk_gb(variant)

        created, offer = self._create_with_fallback(candidates[:4], InstanceSpec(image, label, disk))
        hours_cap = min(cfg.limits.max_session_hours, cfg.limits.max_session_cost / offer.dph_total)
        now = time.time()
        s = Session(
            id=sid, provider=self.provider.name, instance_id=created.id, label=label,
            model_key=spec.key, variant=vname, engine=engine, served_name=spec.served_name, repo=variant.repo,
            context_length=context, tool_calling=spec.tool_calling, output_limit=min(32768, context // 4),
            gpu_name=offer.gpu_name, num_gpus=offer.num_gpus, total_vram_gb=offer.total_vram_gb, dph=offer.dph_total,
            ssh_user=cfg.ssh.user, ssh_key=str(key_path),
            known_hosts=str(key_path.parent / f"known_hosts_{sid}"),
            local_port=cfg.ssh.local_port, remote_port=REMOTE_PORT,
            api_key=secrets.token_urlsafe(32), created_at=now, deadline_epoch=now + hours_cap * 3600,
        )
        self.store.save(s)  # persisted BEFORE anything can fail, so cleanup always has a handle
        self.ui.say(f"Instance {created.id} created.")
        try:
            self._provision(s, spec, variant, pubkey, created.instance_api_key, hf_token, aws_env)
        except BaseException as e:  # includes KeyboardInterrupt
            self.ui.warn(f"Startup failed ({type(e).__name__}); destroying instance {s.instance_id} so nothing keeps billing…")
            self._emergency_cleanup(s)
            raise
        s.phase = "ready"
        self.store.save(s)
        return s

    def _create_with_fallback(self, cands: list[Offer], spec: InstanceSpec):
        last: Exception | None = None
        for offer in cands:
            try:
                return self.provider.create_instance(offer, spec), offer
            except OfferUnavailable as e:
                self.ui.warn(f"Offer {offer.id} gone, trying next…"); last = e
        raise ProvisioningError(f"All top offers were taken before creation: {last}")

    def _provision(self, s: Session, spec: ModelSpec, variant: Variant, pubkey: str,
                   instance_api_key: str | None, hf_token: str | None, aws_env: dict[str, str]) -> None:
        cfg = self.cfg
        t0 = time.monotonic()
        budget = cfg.limits.startup_timeout_minutes * 60

        def remaining() -> float:
            left = budget - (time.monotonic() - t0)
            if left <= 0:
                raise ProvisioningError(f"Startup exceeded the {cfg.limits.startup_timeout_minutes:g} min timeout.")
            return left

        self.ui.say("Waiting for the instance to boot (image pull can take a few minutes)…")
        inst = self._wait_running(s, remaining)
        assert inst.ssh
        s.ssh_host, s.ssh_port = inst.ssh.host, inst.ssh.port
        self.store.save(s)
        self.provider.attach_ssh_key(s.instance_id, pubkey)
        ssh = self.make_ssh(s)
        self.ui.say(f"Connecting over SSH ({s.ssh_host}:{s.ssh_port})…")
        ssh.wait_ready(min(remaining(), 300), sleep=self.sleep)
        remote = self.make_remote(ssh)
        secrets_env = {"CODER_API_KEY": s.api_key, "HF_TOKEN": hf_token or "", **aws_env}
        remote.provision(secrets_env, expect_gpus=s.num_gpus)

        if instance_api_key:
            idle = cfg.idle.timeout_minutes if cfg.idle.enabled else None
            remote.start_watchdog(s.instance_id, instance_api_key, s.deadline_epoch or time.time() + 3600 * cfg.limits.max_session_hours,
                                  idle, REMOTE_PORT, idle_not_before_epoch=time.time() + budget)
        else:
            self.ui.warn("Provider issued no instance-scoped key: the remote self-destruct watchdog is NOT active. Use `coder stop` promptly.")

        target = inference.model_dir(spec, s.variant)
        self.ui.say(f"Downloading {variant.repo} (~{variant.download_gb:g} GB) "
                    f"{'via S3 cache' if cfg.storage.enabled else 'from Hugging Face'}…")
        remote.download_model(variant.repo, target, variant.download_gb, cfg.storage, hf_token=bool(hf_token), timeout_s=remaining())

        tp = inference.pick_gpu_count(variant, s.num_gpus)
        cmd = inference.build_serve_command(spec, s.variant, variant, s.engine, tp, s.context_length)
        self.ui.say(f"Starting {s.engine} (tensor-parallel {tp}, context {s.context_length})…")
        remote.start_server(cmd)
        s.tunnel_pid = self.open_tunnel(s)
        self.store.save(s)
        self.ui.say("Loading the model into GPU memory…")
        remote.wait_healthy(s.base_url, s.api_key, s.served_name, remaining())
        self._apply_opencode(s)

    def _wait_running(self, s: Session, remaining: Callable[[], float]) -> Instance:
        while True:
            remaining()
            inst = self.provider.get_instance(s.instance_id)
            if inst is None:
                raise ProvisioningError("Instance disappeared while booting.")
            if inst.state is InstanceState.FAILED:
                raise ProvisioningError(f"Instance entered state '{inst.raw_status}': {inst.status_msg}")
            if inst.state is InstanceState.RUNNING and inst.ssh:
                return inst
            self.sleep(8)

    def _apply_opencode(self, s: Session) -> str:
        spec = self.registry.get(s.model_key)
        block = opencode.provider_block(
            display_name=spec.display_name, model_id=s.served_name, base_url=s.base_url, api_key=s.api_key,
            context=s.context_length, output=s.output_limit, tool_call=s.tool_calling,
        )
        return opencode.apply(self.cfg.effective_opencode_path, self.cfg.opencode.provider_id, block, s.served_name,
                              set_default_model=self.cfg.opencode.set_default_model)

    @staticmethod
    def _aws_env() -> dict[str, str]:
        env = {k: os.environ.get(k, "") for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")}
        if not env["AWS_ACCESS_KEY_ID"] or not env["AWS_SECRET_ACCESS_KEY"]:
            raise CoderError("storage.enabled requires AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in the environment.")
        return env

    # ----------------------------------------------------------------- teardown
    def _emergency_cleanup(self, s: Session) -> None:
        try:
            self.close_tunnel(s.tunnel_pid)
        except Exception:
            pass
        try:
            opencode.restore()
        except Exception:
            pass
        try:
            self._destroy_and_verify(s)
            self._finish(s, destroyed_confirmed=True)
        except Exception as e:
            s.phase = "destroy_failed"
            self.store.save(s)
            self.ui.warn(
                f"!!! COULD NOT CONFIRM DESTRUCTION of instance {s.instance_id} ({e}). It may still be billing. "
                "Run `coder destroy`, or delete it at https://cloud.vast.ai/instances/ ."
            )

    def _destroy_and_verify(self, s: Session, attempts: int = 6) -> None:
        last: Exception | None = None
        for i in range(attempts):
            try:
                self.provider.destroy_instance(s.instance_id)
                inst = self.provider.get_instance(s.instance_id)
                listed = any(i.id == s.instance_id for i in self.provider.list_managed_instances())
                if (inst is None or inst.state is InstanceState.GONE) and not listed:
                    return
                last = ProvisioningError(f"instance still present (state {inst.state.value if inst else 'listed'})")
            except ProviderError as e:
                last = e
            self.sleep(3 * (i + 1))
        raise ProvisioningError(f"destruction not verified: {last}")

    def _finish(self, s: Session, *, destroyed_confirmed: bool) -> float:
        end = time.time()
        cost = s.cost(end)
        if destroyed_confirmed:
            costs.record_session(s.id, s.created_at, end, cost, s.model_key)
        self.store.clear()
        return cost

    def stop(self, *, yes: bool = False, force: bool = False) -> float | None:
        """Destroy the session's instance (the default meaning of `stop`). Returns the estimated cost."""
        with self.store.lock():
            s = self.store.load()
            if s is None:
                self.ui.say("No active session. (Run `coder cleanup` to look for orphaned instances.)")
                return None
            if not force:
                busy = self.active_requests(s) if self.tunnel_alive(s) else None
                if busy:
                    self.ui.warn(f"{busy} inference request(s) are still running — an agent operation is probably mid-flight.")
                    if not yes and not self.ui.confirm("Destroy anyway?"):
                        raise CoderError("Aborted.")
                elif busy is None and self.tunnel_alive(s):
                    self.ui.warn("Could not read in-flight request count; an agent operation may still be running.")
            if not yes and not self.ui.confirm(
                f"PERMANENTLY destroy instance {s.instance_id} ({s.num_gpus}x {s.gpu_name}) and all data on it? Session cost so far ≈ ${s.cost():.2f}", default=force
            ):
                raise CoderError("Aborted.")
            s.phase = "destroying"
            self.store.save(s)
            if self.close_tunnel(s.tunnel_pid):
                self.ui.say("SSH tunnel closed.")
            try:
                if opencode.restore():
                    self.ui.say("OpenCode configuration restored.")
            except CoderError as e:
                self.ui.warn(f"Could not restore OpenCode config: {e} (run `coder restore-opencode`)")
            try:
                self._destroy_and_verify(s)
            except Exception:
                s.phase = "destroy_failed"
                self.store.save(s)
                raise
            cost = self._finish(s, destroyed_confirmed=True)
            self.ui.say(f"Instance {s.instance_id} destroyed and verified gone.")
            return cost

    def destroy(self, *, yes: bool = False) -> float | None:
        return self.stop(yes=yes, force=True)

    # ------------------------------------------------------------------- status
    def status(self) -> StatusInfo | None:
        s = self.store.load()
        if s is None:
            return None
        info = StatusInfo(session=s, instance=None, cost=s.cost())
        try:
            info.instance = self.provider.get_instance(s.instance_id)
        except CoderError as e:
            info.instance_error = str(e)
        info.tunnel_alive = self.tunnel_alive(s)
        info.server_ready = info.tunnel_alive and self.server_ready(s)
        info.month_cost, info.month_sessions = costs.month_to_date()
        lim = self.cfg.limits
        if info.cost >= lim.max_session_cost:
            info.warnings.append(f"Session cost ${info.cost:.2f} reached the ${lim.max_session_cost:.2f} limit — run `coder stop`.")
        elif info.cost >= lim.warn_at_fraction * lim.max_session_cost:
            info.warnings.append(f"Session cost ${info.cost:.2f} is above {lim.warn_at_fraction:.0%} of the ${lim.max_session_cost:.2f} limit.")
        if s.elapsed_hours() >= lim.max_session_hours:
            info.warnings.append("Max session duration reached; the remote watchdog should be destroying the instance.")
        if info.month_cost + info.cost >= lim.warn_at_fraction * lim.monthly_budget:
            info.warnings.append(f"Month-to-date ≈ ${info.month_cost + info.cost:.2f} of ${lim.monthly_budget:.0f} budget.")
        if info.instance is None and not info.instance_error:
            info.warnings.append("Instance no longer exists at the provider; run `coder destroy` to clear local state.")
        return info

    # --------------------------------------------------------------- reconcile
    def find_orphans(self) -> list[Instance]:
        s = self.store.load()
        keep = s.instance_id if s else None
        return [i for i in self.provider.list_managed_instances() if i.id != keep]

    def destroy_orphans(self, orphans: list[Instance]) -> list[int]:
        destroyed = []
        for i in orphans:
            if not i.label.startswith(MANAGED_LABEL_PREFIX):  # belt and braces: never touch foreign instances
                continue
            self.provider.destroy_instance(i.id)
            destroyed.append(i.id)
        return destroyed
