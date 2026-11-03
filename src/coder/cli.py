"""Typer CLI. Thin: parses arguments, renders output, delegates to the Manager."""
from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Annotated, Optional

import typer
import yaml
from rich.console import Console
from rich.prompt import Confirm, FloatPrompt, Prompt
from rich.table import Table

from . import costs, opencode, paths
from .config import Config, dump_config, keychain_set_interactive, load_config, load_dotenv
from .errors import CoderError
from .lifecycle import Manager
from .models import load_registry
from .offers import find_offers
from .providers import build_provider
from .session import SessionStore

app = typer.Typer(help="On-demand ephemeral cloud GPUs for local AI coding agents.", no_args_is_help=True, add_completion=False)
console = Console()
err = Console(stderr=True)
YES = Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation prompts.")]


class RichUI:
    def say(self, msg: str) -> None:
        console.print(f"[cyan]•[/cyan] {msg}")

    def warn(self, msg: str) -> None:
        console.print(f"[yellow]![/yellow] {msg}")

    def confirm(self, msg: str, *, default: bool = False) -> bool:
        return Confirm.ask(msg, default=default, console=console)

    def show_plan(self, offer, est, model, variant_name, variant, engine, context) -> None:
        t = Table(title=f"{model.display_name} ({variant_name}, {variant.quant}) on {engine}", show_header=False)
        t.add_row("GPU", f"{offer.num_gpus}x {offer.gpu_name}  ({offer.total_vram_gb:.0f} GB VRAM total)")
        t.add_row("Location / reliability", f"{offer.geolocation} / {offer.reliability:.3f}")
        t.add_row("Price", f"${offer.dph_total:.2f}/hour")
        t.add_row("Startup (est.)", f"{est.startup_minutes:.0f} min ≈ ${est.startup_cost:.2f}")
        t.add_row(f"{est.session_hours:g} h session (est.)", f"${est.session_cost:.2f} total")
        t.add_row("Monthly @ 22 days (est.)", f"${est.monthly_cost:.0f}")
        console.print(t)


def _load() -> Config:
    logging.basicConfig(level=logging.WARNING)
    load_dotenv()
    return load_config()


def _manager(cfg: Config) -> Manager:
    return Manager(cfg, build_provider(cfg), load_registry(), SessionStore(), RichUI())


def _guard(fn):
    """Show CoderError as a clean message, everything else as a real traceback."""
    import functools

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except CoderError as e:
            err.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    return wrapper


@app.command()
@_guard
def start(
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Model key (see `coder models`).")] = None,
    variant: Annotated[Optional[str], typer.Option(help="Checkpoint variant, e.g. fp8 or bf16.")] = None,
    engine: Annotated[Optional[str], typer.Option(help="vllm or sglang (default: from config).")] = None,
    context: Annotated[Optional[int], typer.Option(help="Context length in tokens.")] = None,
    offer: Annotated[Optional[int], typer.Option("--offer", "-o", min=1, help="Rent row N of `coder offers` instead of the top one.")] = None,
    yes: YES = False,
) -> None:
    """Rent a GPU, load the model, open the tunnel and wire up OpenCode."""
    mgr = _manager(_load())
    s, reused = mgr.start(model, variant=variant, engine=engine, context=context, yes=yes, offer_rank=offer)
    console.print(f"\n[bold green]{'Reusing existing session' if reused else 'Ready'}.[/bold green]")
    console.print(f"  Model     {s.served_name}  ({s.engine}, {s.context_length} ctx)")
    console.print(f"  Endpoint  {s.base_url}  (SSH tunnel, bearer token required)")
    console.print(f"  OpenCode  select [bold]{mgr.cfg.opencode.provider_id}/{s.served_name}[/bold] (set as default); run `opencode` in your project")
    console.print(f"  Cost      ${s.dph:.2f}/h — run [bold]coder stop[/bold] when done (auto-destroy at {time.strftime('%H:%M', time.localtime(s.deadline_epoch))})")


@app.command()
@_guard
def stop(yes: YES = False) -> None:
    """Destroy the GPU instance (and its disk). `stop` here always means destroy."""
    cost = _manager(_load()).stop(yes=yes)
    if cost is not None:
        console.print(f"[bold]Session cost (estimate): ${cost:.2f}[/bold]")


@app.command()
@_guard
def destroy(yes: YES = False) -> None:
    """Destroy all resources of the current session without graceful checks."""
    cost = _manager(_load()).destroy(yes=yes)
    if cost is not None:
        console.print(f"[bold]Session cost (estimate): ${cost:.2f}[/bold]")


@app.command()
@_guard
def status() -> None:
    """Show the current session."""
    info = _manager(_load()).status()
    if info is None:
        console.print("No active session.")
        return
    s = info.session
    st = info.instance.state.value if info.instance else ("unknown (provider error)" if info.instance_error else "gone")
    t = Table(show_header=False, title=f"Session {s.id} [{s.phase}]")
    t.add_row("Provider", s.provider)
    t.add_row("Instance", str(s.instance_id))
    t.add_row("GPU", f"{s.num_gpus}x {s.gpu_name}")
    t.add_row("VRAM", f"{s.total_vram_gb:.0f} GB")
    t.add_row("Model", f"{s.served_name} ({s.repo}, {s.engine})")
    t.add_row("Instance state", st)
    t.add_row("Inference server", "ready" if info.server_ready else "not reachable")
    t.add_row("SSH tunnel", f"up (pid {s.tunnel_pid}, {s.base_url})" if info.tunnel_alive else "down")
    h = s.elapsed_hours()
    t.add_row("Duration", f"{int(h)}h{int(h % 1 * 60):02d}m")
    t.add_row("Cost so far (est.)", f"${info.cost:.2f} at ${s.dph:.2f}/h")
    t.add_row("Month to date (est.)", f"${info.month_cost + info.cost:.2f}")
    console.print(t)
    for w in info.warnings:
        console.print(f"[yellow]![/yellow] {w}")
    if not info.tunnel_alive and s.phase == "ready":
        console.print("Tunnel is down (Mac sleep / network?). Run `coder start` to re-open it.")


@app.command()
@_guard
def offers(
    model: Annotated[Optional[str], typer.Option("--model", "-m")] = None,
    variant: Annotated[Optional[str], typer.Option()] = None,
    top: Annotated[int, typer.Option(help="Rows to show.")] = 15,
) -> None:
    """List compatible GPU offers, best value first."""
    cfg = _load()
    reg = load_registry()
    spec = reg.get(model or cfg.model)
    vname, v = spec.variant(variant or cfg.variant)
    provider = build_provider(cfg)
    try:
        found = find_offers(provider, cfg, spec, v)
    finally:
        provider.close()
    if not found:
        console.print("No matching offers. Try raising limits.max_hourly_price or relaxing offers.* in your config.")
        return
    t = Table(title=f"{spec.display_name} / {vname}: {len(found)} offers")
    for c in ("#", "GPUs", "VRAM", "$/h", "Reliab.", "Down Mb/s", "Where", "Est. startup", f"Est. {cfg.limits.expected_session_hours:g}h"):
        t.add_column(c)
    for i, o in enumerate(found[:top], 1):
        e = costs.estimate(o, v.download_gb, cfg.limits.expected_session_hours)
        t.add_row(str(i), f"{o.num_gpus}x {o.gpu_name}", f"{o.total_vram_gb:.0f} GB", f"{o.dph_total:.2f}", f"{o.reliability:.3f}",
                  f"{o.inet_down_mbps:.0f}", o.geolocation, f"{e.startup_minutes:.0f} min", f"${e.session_cost:.2f}")
    console.print(t)


@app.command()
@_guard
def models(details: Annotated[bool, typer.Option("--details", help="Also show repos, engines, parsers and notes.")] = False) -> None:
    """List registered models and their hardware requirements."""
    t = Table(title="Model registry")
    t.add_column("Key", no_wrap=True)
    for c in ("Variant", "Quant", "Size", "Min VRAM", "GPUs", "Context"):
        t.add_column(c, no_wrap=True)
    for m in load_registry():
        for vn, v in m.variants.items():
            star = "*" if vn == m.default_variant else ""
            t.add_row(m.key, vn + star, v.quant, f"~{v.download_gb:g}GB", f"{v.min_total_vram_gb:g}GB", "/".join(map(str, v.gpus)), f"{m.context_length // 1024}K")
    console.print(t)
    console.print("* default variant. Sizes/VRAM are estimates (min VRAM = total across GPUs; >= per-GPU minimum applies too).")
    if details:
        for m in load_registry():
            console.print(f"\n[bold]{m.key}[/bold] — {m.description}")
            for vn, v in m.variants.items():
                console.print(f"  {vn}: {v.repo} | engines {', '.join(v.engines)} | tool parser {v.tool_call_parser} | min {v.min_gpu_vram_gb:g}GB/GPU")
                if v.notes:
                    console.print(f"     [dim]{v.notes}[/dim]")


@app.command()
@_guard
def logs(
    lines: Annotated[int, typer.Option("-n", help="Number of lines.")] = 100,
    follow: Annotated[bool, typer.Option("--follow", "-f")] = False,
) -> None:
    """Show inference server logs from the instance."""
    cfg = _load()
    s = SessionStore().load()
    if s is None or not s.ssh_host:
        raise CoderError("No active session with a reachable instance.")
    from .remote import HOME
    ssh = Manager(cfg, None, None, SessionStore(), RichUI()).make_ssh(s)  # type: ignore[arg-type]
    if follow:
        cmd = ssh._base() + [f"tail -n {int(lines)} -f {HOME}/server.log"]
        subprocess.run(cmd)
    else:
        console.print(ssh.run(f"tail -n {int(lines)} {HOME}/server.log", check=False).stdout)


@app.command()
@_guard
def cleanup(
    destroy_orphans: Annotated[bool, typer.Option("--destroy", help="Destroy the orphans found.")] = False,
    yes: YES = False,
) -> None:
    """Find (and optionally destroy) coder-tagged instances that no local session owns."""
    mgr = _manager(_load())
    orphans = mgr.find_orphans()
    if not orphans:
        console.print("[green]No orphaned coder instances.[/green]")
        return
    t = Table(title="Orphaned instances (label starts with 'coder-managed')")
    for c in ("ID", "Label", "State", "GPU", "$/h"):
        t.add_column(c)
    for i in orphans:
        t.add_row(str(i.id), i.label, i.state.value, f"{i.num_gpus}x {i.gpu_name}", f"{i.dph_total:.2f}")
    console.print(t)
    if not destroy_orphans:
        console.print("Re-run with --destroy to remove them. Instances without the coder label are never touched.")
        return
    if yes or Confirm.ask(f"Permanently destroy {len(orphans)} instance(s)?", default=False):
        done = mgr.destroy_orphans(orphans)
        console.print(f"Destroyed: {', '.join(map(str, done))}")


@app.command("restore-opencode")
@_guard
def restore_opencode() -> None:
    """Put back your previous OpenCode provider/model settings."""
    console.print("OpenCode configuration restored." if opencode.restore() else "Nothing to restore.")


@app.command()
@_guard
def config(
    init: Annotated[bool, typer.Option("--init", help="Interactive setup; writes the config file.")] = False,
    store_key: Annotated[Optional[str], typer.Option("--store-key", help="Store a secret in the macOS Keychain: vast | hf.")] = None,
) -> None:
    """Show the effective configuration, or set it up interactively."""
    if store_key:
        if store_key not in ("vast", "hf"):
            raise CoderError("--store-key must be 'vast' or 'hf'")
        keychain_set_interactive(store_key)
        console.print(f"Stored in Keychain as 'coder-{store_key}'.")
        return
    if init:
        cfg = load_config()
        reg = load_registry()
        cfg.model = Prompt.ask("Default model", choices=reg.keys(), default=cfg.model, console=console)
        cfg.limits.max_hourly_price = FloatPrompt.ask("Max $/hour", default=cfg.limits.max_hourly_price, console=console)
        cfg.limits.max_session_cost = FloatPrompt.ask("Max $ per session", default=cfg.limits.max_session_cost, console=console)
        cfg.limits.monthly_budget = FloatPrompt.ask("Monthly budget $", default=cfg.limits.monthly_budget, console=console)
        cfg.idle.enabled = Confirm.ask("Enable remote idle auto-destroy (see README limits)?", default=cfg.idle.enabled, console=console)
        if cfg.idle.enabled:
            cfg.idle.timeout_minutes = FloatPrompt.ask("Idle minutes", default=cfg.idle.timeout_minutes, console=console)
        cfg.storage.enabled = Confirm.ask("Use an S3 bucket as model cache?", default=cfg.storage.enabled, console=console)
        if cfg.storage.enabled:
            cfg.storage.bucket = Prompt.ask("Bucket", default=cfg.storage.bucket, console=console)
            ep = Prompt.ask("Endpoint URL (blank for AWS)", default=cfg.storage.endpoint_url or "", console=console)
            cfg.storage.endpoint_url = ep or None
        p = paths.config_file()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(dump_config(cfg))
        console.print(f"Wrote {p}. API keys are never written here: use VAST_API_KEY or `coder config --store-key vast`.")
        return
    cfg = load_config()
    if used := load_dotenv():
        console.print(f"[dim]env files: {', '.join(map(str, used))} (secrets are not displayed)[/dim]")
    console.print(f"[dim]{paths.config_file()}{'' if paths.config_file().exists() else ' (not present, showing defaults)'}[/dim]")
    console.print(yaml.safe_dump(cfg.model_dump(mode="json", exclude_none=True), sort_keys=False))


def main() -> None:  # pragma: no cover
    app()
