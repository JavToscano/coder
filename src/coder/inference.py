"""Inference-server command construction and client-side probes (through the tunnel)."""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

import httpx

from .errors import ConfigError
from .models import ModelSpec, Variant

REMOTE_PORT = 8000
API_KEY_ENV = "CODER_API_KEY"  # expanded by the remote shell from a 0600 env file, never in our argv
MODEL_DIR = "/root/models"


def pick_engine(variant: Variant, requested: str) -> str:
    if requested == "auto":
        return variant.engines[0]
    if requested not in variant.engines:
        raise ConfigError(
            f"Engine '{requested}' is not documented for this checkpoint (supported: {', '.join(variant.engines)})."
        )
    return requested


def pick_gpu_count(variant: Variant, num_gpus: int) -> int:
    """Tensor-parallel size = GPUs on the box (offers are filtered to allowed counts)."""
    if num_gpus not in variant.gpus:
        raise ConfigError(f"{num_gpus} GPUs is not an allowed tensor-parallel size {variant.gpus} for this checkpoint")
    return num_gpus


def model_dir(spec: ModelSpec, variant_name: str) -> str:
    return f"{MODEL_DIR}/{spec.key}-{variant_name}"


@dataclass(frozen=True)
class ServeCommand:
    argv: list[str]
    api_key_flag: list[str]  # contains a shell expansion, so it must not be shlex-quoted

    def shell(self) -> str:
        return " ".join(shlex.quote(a) for a in self.argv + []) + " " + " ".join(self.api_key_flag)


def build_serve_command(
    spec: ModelSpec, variant_name: str, variant: Variant, engine: str, tp: int, context: int
) -> ServeCommand:
    path = model_dir(spec, variant_name)
    api_flag = ["--api-key", f'"${API_KEY_ENV}"']
    extra = list(variant.extra_args.get(engine, []))
    if engine == "vllm":
        argv = [
            "vllm", "serve", path,
            "--host", "127.0.0.1", "--port", str(REMOTE_PORT),
            "--served-model-name", spec.served_name,
            "--tensor-parallel-size", str(tp),
            "--max-model-len", str(context),
        ]
        if spec.tool_calling and variant.tool_call_parser:
            argv += ["--enable-auto-tool-choice", "--tool-call-parser", variant.tool_call_parser]
        if variant.reasoning_parser:
            argv += ["--reasoning-parser", variant.reasoning_parser]
    elif engine == "sglang":
        argv = [
            "python3", "-m", "sglang.launch_server",
            "--model-path", path,
            "--host", "127.0.0.1", "--port", str(REMOTE_PORT),
            "--served-model-name", spec.served_name,
            "--tp-size", str(tp),
            "--context-length", str(context),
            "--enable-metrics",
        ]
        if spec.tool_calling and variant.tool_call_parser:
            argv += ["--tool-call-parser", variant.tool_call_parser]
        if variant.reasoning_parser:
            argv += ["--reasoning-parser", variant.reasoning_parser]
    else:  # pragma: no cover
        raise ConfigError(f"Unsupported engine {engine}")
    return ServeCommand(argv + extra, api_flag)


def serve_script(cmd: ServeCommand) -> str:
    return (
        "#!/usr/bin/env bash\n"
        "set -a; . /root/coder/secrets.env; set +a\n"
        "export HF_HUB_OFFLINE=1\n"  # weights are already local; never phone home at serve time
        "export PATH=$PATH:/usr/local/bin:/opt/venv/bin\n"
        f"exec {cmd.shell()}\n"
    )


# --- probes ---------------------------------------------------------------

def models_ready(base_url: str, api_key: str, served_name: str, timeout: float = 5.0) -> bool:
    """True once the OpenAI-compatible server lists the model (authenticated, via the tunnel)."""
    try:
        r = httpx.get(f"{base_url}/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)
    except httpx.HTTPError:
        return False
    if r.status_code != 200:
        return False
    try:
        return any(m.get("id") == served_name for m in r.json().get("data", []))
    except ValueError:
        return False


_RUNNING = re.compile(r"^(?:vllm:num_requests_running|sglang:num_running_reqs)(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", re.M)
_WAITING = re.compile(r"^(?:vllm:num_requests_waiting|sglang:num_queue_reqs)(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", re.M)


def parse_active_requests(metrics_text: str) -> int | None:
    """Running + queued requests from Prometheus text, or None if the gauges are absent."""
    run = _RUNNING.findall(metrics_text)
    if not run:
        return None
    wait = _WAITING.findall(metrics_text)
    return int(sum(float(x) for x in run) + sum(float(x) for x in wait))


def active_requests(local_port: int, timeout: float = 5.0) -> int | None:
    try:
        r = httpx.get(f"http://127.0.0.1:{local_port}/metrics", timeout=timeout)
        return parse_active_requests(r.text) if r.status_code == 200 else None
    except httpx.HTTPError:
        return None
