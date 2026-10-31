"""Everything that happens on the GPU instance: environment checks, model download,
server launch/health, watchdog, logs. All secrets travel over ssh stdin into 0600 files."""
from __future__ import annotations

import json
import shlex
import time
from importlib import resources
from typing import Callable

from .config import StorageConfig
from .errors import ProvisioningError, SSHError
from .inference import ServeCommand, models_ready, serve_script
from .ssh import SSHClient

HOME = "/root/coder"
MANIFEST = ".coder-manifest.json"


def _dotenv(values: dict[str, str]) -> str:
    return "".join(f"{k}={shlex.quote(v)}\n" for k, v in values.items() if v)


class Remote:
    def __init__(self, ssh: SSHClient, log: Callable[[str], None] = lambda m: None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.ssh, self.log, self._sleep = ssh, log, sleep

    # -- environment -----------------------------------------------------------
    def provision(self, secrets: dict[str, str], *, expect_gpus: int) -> None:
        self.ssh.run(f"mkdir -p {HOME} /root/models && chmod 700 {HOME}")
        self.ssh.write_file(f"{HOME}/secrets.env", _dotenv(secrets))
        r = self.ssh.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", timeout=60)
        n = len([ln for ln in r.stdout.splitlines() if ln.strip()])
        if n < expect_gpus:
            raise ProvisioningError(f"Instance exposes {n} GPU(s) but {expect_gpus} were expected:\n{r.stdout}")
        # Fail fast if the image lacks the engine (e.g. wrong image tag).
        self.ssh.run("export PATH=$PATH:/usr/local/bin:/opt/venv/bin; command -v vllm || python3 -c 'import sglang'", check=True, timeout=60)

    # -- model download ---------------------------------------------------------
    def download_model(
        self, repo: str, target: str, expected_gb: float, storage: StorageConfig | None,
        *, hf_token: bool, timeout_s: float, retries: int = 3,
    ) -> None:
        use_s3 = bool(storage and storage.enabled)
        s3 = f"s3://{storage.bucket}/{storage.prefix}/{repo}" if use_s3 and storage else ""
        script = self._download_script(repo, target, s3, storage, hf_token)
        self.ssh.write_file(f"{HOME}/download.sh", script, mode="700")
        deadline = time.monotonic() + timeout_s
        for attempt in range(1, retries + 1):
            self.ssh.run(f"rm -f {HOME}/download.rc; nohup setsid bash {HOME}/download.sh > {HOME}/download.log 2>&1 < /dev/null & echo started")
            rc = self._wait_download(target, expected_gb, deadline)
            if rc == 0:
                return
            tail = self.ssh.run(f"tail -n 15 {HOME}/download.log", check=False).stdout
            self.log(f"download attempt {attempt}/{retries} failed (exit {rc}); retrying (resumable)")
            if attempt == retries:
                raise ProvisioningError(f"Model download failed after {retries} attempts:\n{tail}")
            self._sleep(10)

    def _wait_download(self, target: str, expected_gb: float, deadline: float) -> int:
        last_report = 0.0
        while time.monotonic() < deadline:
            self._sleep(10)
            try:
                r = self.ssh.run(
                    f"echo RC=$(cat {HOME}/download.rc 2>/dev/null); echo MB=$(du -sm {target} 2>/dev/null | cut -f1)",
                    check=False, timeout=45,
                )
            except SSHError:
                continue  # transient network blip; the download runs detached on the host
            fields = dict(ln.split("=", 1) for ln in r.stdout.split() if "=" in ln)
            mb = int(fields["MB"]) if fields.get("MB", "").isdigit() else 0
            if time.monotonic() - last_report > 30:
                self.log(f"downloaded {mb / 1000:.1f} / ~{expected_gb:.0f} GB")
                last_report = time.monotonic()
            if fields.get("RC", "").lstrip("-").isdigit():
                return int(fields["RC"])
        raise ProvisioningError("Model download timed out")

    @staticmethod
    def _download_script(repo: str, target: str, s3: str, storage: StorageConfig | None, hf_token: bool) -> str:
        hf = (
            "python3 - <<'PY'\n"
            "import os\n"
            "from huggingface_hub import snapshot_download\n"
            f"snapshot_download(repo_id={repo!r}, local_dir={target!r}, token=os.environ.get('HF_TOKEN') or None, max_workers=8)\n"
            "PY\n"
        )
        head = f"#!/usr/bin/env bash\nset -uo pipefail\nset -a; . {HOME}/secrets.env; set +a\nmkdir -p {shlex.quote(target)}\n"
        if not s3 or not storage:
            return head + f"( set -e\n{hf})\necho $? > {HOME}/download.rc\n"
        endpoint = f"--endpoint-url {shlex.quote(storage.endpoint_url)} " if storage.endpoint_url else ""
        region = f"--region {shlex.quote(storage.region)} " if storage.region else ""
        aws = f"aws {endpoint}{region}"
        # Strategy B: bucket first (sync only transfers changed files; no --delete, ever, in
        # either direction). On a miss, fetch from HF and publish to the bucket for next time.
        verify = (
            "python3 - <<'PY'\n"
            "import json, os, sys\n"
            f"t = {target!r}\n"
            f"m = json.load(open(os.path.join(t, {MANIFEST!r})))\n"
            "bad = [f for f, size in m.items() if not os.path.isfile(os.path.join(t, f)) or os.path.getsize(os.path.join(t, f)) != size]\n"
            "sys.exit('integrity check failed: ' + ', '.join(bad[:5]) if bad else 0)\n"
            "PY\n"
        )
        write_manifest = (
            "python3 - <<'PY'\n"
            "import json, os\n"
            f"t = {target!r}\n"
            "m = {}\n"
            "for root, _, files in os.walk(t):\n"
            "    for f in files:\n"
            "        p = os.path.join(root, f)\n"
            f"        if f != {MANIFEST!r} and '.cache' not in p: m[os.path.relpath(p, t)] = os.path.getsize(p)\n"
            f"json.dump(m, open(os.path.join(t, {MANIFEST!r}), 'w'))\n"
            "PY\n"
        )
        return head + (
            "(\nset -e\n"
            "command -v aws >/dev/null || pip install -q awscli\n"
            f"if {aws}s3 ls {shlex.quote(s3 + '/' + MANIFEST)} >/dev/null 2>&1; then\n"
            f"  echo 'model cache hit in bucket'\n"
            f"  {aws}s3 sync {shlex.quote(s3)} {shlex.quote(target)} --only-show-errors\n"
            f"{verify}"
            "else\n"
            "  echo 'model cache miss; downloading from Hugging Face'\n"
            f"{hf}{write_manifest}"
            f"  {aws}s3 sync {shlex.quote(target)} {shlex.quote(s3)} --only-show-errors --exclude '.cache/*'\n"
            "fi\n"
            f")\necho $? > {HOME}/download.rc\n"
        )

    # -- server -------------------------------------------------------------------
    def start_server(self, cmd: ServeCommand) -> None:
        self.ssh.write_file(f"{HOME}/serve.sh", serve_script(cmd), mode="700")
        self.ssh.run(f"nohup setsid bash {HOME}/serve.sh > {HOME}/server.log 2>&1 < /dev/null & echo started")

    def server_alive(self) -> bool:
        r = self.ssh.run("pgrep -f 'vllm serve|sglang.launch_server' >/dev/null && echo up || echo down", check=False, timeout=45)
        return "up" in r.stdout

    def wait_healthy(self, base_url: str, api_key: str, served_name: str, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if models_ready(base_url, api_key, served_name):
                return
            try:
                if not self.server_alive():
                    raise ProvisioningError("Inference server exited during startup.\n" + self.logs(40))
            except SSHError:
                pass
            self._sleep(8)
        raise ProvisioningError(f"Inference server not ready within {int(timeout_s)}s.\n" + self.logs(40))

    def logs(self, lines: int = 100) -> str:
        r = self.ssh.run(f"tail -n {int(lines)} {HOME}/server.log 2>&1", check=False, timeout=45)
        return r.stdout

    # -- watchdog -------------------------------------------------------------------
    def start_watchdog(self, instance_id: int, instance_api_key: str, deadline_epoch: float, idle_minutes: float | None,
                       port: int, idle_not_before_epoch: float) -> None:
        script = resources.files("coder").joinpath("data/watchdog.py").read_text()
        self.ssh.write_file(f"{HOME}/watchdog.py", script, mode="700")
        self.ssh.write_file(f"{HOME}/provider.key", instance_api_key)
        cfg = {"instance_id": instance_id, "deadline_epoch": deadline_epoch, "idle_minutes": idle_minutes,
               "port": port, "idle_not_before_epoch": idle_not_before_epoch}
        self.ssh.write_file(f"{HOME}/watchdog.json", json.dumps(cfg))
        self.ssh.run(f"nohup setsid python3 {HOME}/watchdog.py > {HOME}/watchdog.log 2>&1 < /dev/null & echo started")
