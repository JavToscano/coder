"""SSH execution and local port-forward tunnels via the system `ssh` binary."""
from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Callable, Sequence

from .errors import SSHError

Runner = Callable[..., subprocess.CompletedProcess]


def generate_keypair(key_path: Path) -> str:
    """Create a dedicated ed25519 key (no passphrase, used only for ephemeral instances).
    Returns the public key. Never touches the user's own ~/.ssh keys."""
    key_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not key_path.exists():
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "coder-ephemeral", "-f", str(key_path)],
            check=True,
        )
    return Path(str(key_path) + ".pub").read_text().strip()


class SSHClient:
    def __init__(
        self, host: str, port: int, user: str, key: Path, known_hosts: Path,
        *, connect_timeout: int = 15, runner: Runner = subprocess.run,
    ) -> None:
        self.host, self.port, self.user = host, port, user
        self.key, self.known_hosts = key, known_hosts
        self.connect_timeout = connect_timeout
        self._run = runner

    def _base(self) -> list[str]:
        self.known_hosts.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        return [
            "ssh", "-i", str(self.key), "-p", str(self.port),
            "-o", "IdentitiesOnly=yes",
            "-o", "BatchMode=yes",
            # Per-session known_hosts: a new instance has a new host key. First contact is
            # trust-on-first-use; later connections in the session are pinned (see README).
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={self.known_hosts}",
            "-o", f"ConnectTimeout={self.connect_timeout}",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
            "-o", "LogLevel=ERROR",
            "-o", "IdentityAgent=none",
            f"{self.user}@{self.host}",
        ]

    def run(self, command: str, *, stdin: str | None = None, timeout: float = 120, check: bool = True) -> subprocess.CompletedProcess:
        """Run a command remotely. Pass secrets via `stdin`, never inside `command`."""
        try:
            r = self._run(self._base() + [command], input=stdin, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise SSHError(f"ssh command timed out after {timeout}s: {command[:80]}") from e
        if check and r.returncode != 0:
            raise SSHError(f"remote command failed (exit {r.returncode}): {command[:80]}\n{r.stderr.strip()[-500:]}")
        return r

    def write_file(self, remote_path: str, content: str, mode: str = "600") -> None:
        self.run(f"umask 077 && mkdir -p \"$(dirname {remote_path})\" && cat > {remote_path} && chmod {mode} {remote_path}", stdin=content)

    def wait_ready(self, timeout_s: float, *, sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> None:
        deadline = clock() + timeout_s
        last = ""
        while clock() < deadline:
            r = self.run("true", timeout=self.connect_timeout + 10, check=False)
            if r.returncode == 0:
                return
            last = r.stderr.strip()[-200:]
            sleep(5)
        raise SSHError(f"SSH to {self.host}:{self.port} not reachable within {int(timeout_s)}s. Last error: {last}")

    def start_tunnel(self, local_port: int, remote_port: int) -> int:
        cmd = self._base()[:-1] + [
            "-N", "-o", "ExitOnForwardFailure=yes",
            "-L", f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
            f"{self.user}@{self.host}",
        ]
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
        time.sleep(1.5)
        if p.poll() is not None:
            err = p.stderr.read().decode(errors="replace") if p.stderr else ""
            raise SSHError(f"SSH tunnel exited immediately: {err.strip()[-300:]}")
        return p.pid


def pid_is_ssh(pid: int | None) -> bool:
    """True only if `pid` is alive AND is an ssh process (guards against PID reuse)."""
    if not pid:
        return False
    r = subprocess.run(["ps", "-p", str(pid), "-o", "comm="], capture_output=True, text=True)
    return r.returncode == 0 and os.path.basename(r.stdout.strip()) == "ssh"


def stop_tunnel(pid: int | None) -> bool:
    if not pid_is_ssh(pid):
        return False
    try:
        os.kill(pid, signal.SIGTERM)  # type: ignore[arg-type]
    except ProcessLookupError:
        return False
    return True
