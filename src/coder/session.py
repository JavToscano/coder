"""Persistent session state, so `status`/`stop` work after the terminal or CLI dies."""
from __future__ import annotations

import fcntl
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal

from pydantic import BaseModel, Field

from . import paths

Phase = Literal["provisioning", "ready", "destroying", "destroy_failed"]


class Session(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    provider: str
    instance_id: int
    label: str
    phase: Phase = "provisioning"
    created_at: float = Field(default_factory=time.time)
    model_key: str
    variant: str
    engine: str
    served_name: str
    repo: str
    context_length: int
    tool_calling: bool = True
    output_limit: int = 32768
    gpu_name: str = ""
    num_gpus: int = 0
    total_vram_gb: float = 0
    dph: float = 0.0
    ssh_host: str = ""
    ssh_port: int = 0
    ssh_user: str = "root"
    ssh_key: str = ""
    known_hosts: str = ""
    local_port: int = 18000
    remote_port: int = 8000
    tunnel_pid: int | None = None
    api_key: str = ""  # per-session bearer token for the inference server
    deadline_epoch: float | None = None  # hard stop enforced by the remote watchdog

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.local_port}/v1"

    def elapsed_hours(self, now: float | None = None) -> float:
        return max(0.0, ((now or time.time()) - self.created_at) / 3600)

    def cost(self, now: float | None = None) -> float:
        return self.elapsed_hours(now) * self.dph


class SessionStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or paths.session_file()

    def load(self) -> Session | None:
        if not self.path.exists():
            return None
        return Session.model_validate_json(self.path.read_text())

    def save(self, s: Session) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # holds the api_key
        with os.fdopen(fd, "w") as f:
            f.write(s.model_dump_json(indent=2))
        os.replace(tmp, self.path)

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)

    @contextmanager
    def lock(self) -> Iterator[None]:
        """Serialises start/stop/destroy across terminals."""
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with open(self.path.with_suffix(".lock"), "w") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                from .errors import CoderError
                raise CoderError("Another coder command is already running (session lock held).") from None
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
