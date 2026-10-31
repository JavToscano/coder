"""User configuration (YAML) and credential lookup (env vars or macOS Keychain)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from . import paths
from .errors import ConfigError, CredentialsError


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OfferConfig(_Strict):
    preferred_gpus: list[str] = Field(default_factory=lambda: ["H100", "H200", "A100", "RTX 6000Ada", "L40S"])
    min_vram_gb: float = 0  # per GPU, on top of the model's own requirement
    min_system_ram_gb: float = 32
    min_reliability: float = Field(0.97, ge=0, le=1)
    min_inet_down_mbps: float = 200
    countries: list[str] = Field(default_factory=list)  # ISO codes, empty = anywhere


class LimitsConfig(_Strict):
    max_hourly_price: float = Field(3.0, gt=0)
    max_session_cost: float = Field(10.0, gt=0)
    max_session_hours: float = Field(6.0, gt=0)
    startup_timeout_minutes: float = Field(35.0, gt=0)
    warn_at_fraction: float = Field(0.8, gt=0, le=1)
    monthly_budget: float = Field(75.0, gt=0)
    expected_session_hours: float = Field(3.0, gt=0)  # used for cost projections only


class IdleConfig(_Strict):
    # Opt-in: the remote watchdog destroys the instance after this many minutes without
    # inference requests. See README "Idle shutdown" for the limits of request tracking.
    enabled: bool = False
    timeout_minutes: float = Field(20.0, gt=0)


class SSHConfig(_Strict):
    user: str = "root"
    key_path: Path | None = None  # default: a dedicated key under the state dir
    local_port: int = Field(18000, ge=1024, le=65535)
    connect_timeout_s: int = 15


class StorageConfig(_Strict):
    enabled: bool = False
    bucket: str = ""
    prefix: str = "coder-models"
    endpoint_url: str | None = None  # for non-AWS S3-compatible stores
    region: str | None = None

    @field_validator("prefix")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip("/")


class OpenCodeConfig(_Strict):
    config_path: Path | None = None
    provider_id: str = "coder"
    set_default_model: bool = True


class Config(_Strict):
    provider: str = "vast"
    model: str = "qwen3-coder-next"
    variant: str | None = None
    engine: Literal["auto", "vllm", "sglang"] = "auto"
    context_length: int | None = Field(None, gt=0)
    images: dict[str, str] = Field(
        default_factory=lambda: {"vllm": "vllm/vllm-openai:latest", "sglang": "lmsysorg/sglang:latest"}
    )
    offers: OfferConfig = Field(default_factory=OfferConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    idle: IdleConfig = Field(default_factory=IdleConfig)
    ssh: SSHConfig = Field(default_factory=SSHConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    opencode: OpenCodeConfig = Field(default_factory=OpenCodeConfig)

    @property
    def effective_ssh_key(self) -> Path:
        return self.ssh.key_path or paths.ssh_dir() / "id_ed25519"

    @property
    def effective_opencode_path(self) -> Path:
        return self.opencode.config_path or paths.opencode_config_file()


def load_config(path: Path | None = None) -> Config:
    path = path or paths.config_file()
    if not path.exists():
        return Config()
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: invalid YAML: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    try:
        return Config.model_validate(raw)
    except ValidationError as e:
        lines = [f"  {'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()]
        raise ConfigError(f"{path}: invalid configuration:\n" + "\n".join(lines)) from e


def dump_config(cfg: Config) -> str:
    return yaml.safe_dump(cfg.model_dump(mode="json", exclude_none=True), sort_keys=False)


# --- credentials -----------------------------------------------------------

KEYCHAIN_SERVICE_PREFIX = "coder-"


def keychain_get(name: str) -> str | None:
    if sys.platform != "darwin":
        return None
    r = subprocess.run(
        ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE_PREFIX + name, "-w"],
        capture_output=True, text=True,
    )
    return r.stdout.strip() or None if r.returncode == 0 else None


def keychain_set_interactive(name: str, account: str = "coder") -> None:
    """Store a secret in the login keychain. `-w` as the last argument makes `security`
    prompt for the value on the terminal, so the secret never appears in argv."""
    subprocess.run(
        ["security", "add-generic-password", "-U", "-a", account, "-s", KEYCHAIN_SERVICE_PREFIX + name, "-w"],
        check=True,
    )


def get_secret(env_var: str, keychain_name: str, *, required: bool, hint: str = "") -> str | None:
    value = os.environ.get(env_var) or keychain_get(keychain_name)
    if value is None and required:
        raise CredentialsError(
            f"{env_var} is not set and no '{KEYCHAIN_SERVICE_PREFIX}{keychain_name}' Keychain item exists. {hint}".strip()
        )
    return value


def vast_api_key() -> str:
    key = get_secret(
        "VAST_API_KEY", "vast", required=True,
        hint="Create a key at https://cloud.vast.ai/manage-keys/ then `export VAST_API_KEY=...` or run `coder config --store-key vast`.",
    )
    assert key
    return key
