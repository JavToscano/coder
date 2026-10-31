# coder

Rent a cloud GPU only while you code, serve a large coding LLM on it, point OpenCode at it through an SSH tunnel, then **destroy everything** when you stop. Nothing persists, so there is no storage bill between sessions.

```
coder start --model qwen3-coder-next   # find offer → confirm → boot → download → serve → tunnel → OpenCode
opencode                               # use it from your Mac, project files stay local
coder stop                             # tunnel closed, OpenCode config restored, instance DESTROYED + verified
```

> **Status: a full start → OpenCode → stop session has run on real Vast.ai.** Confirmed by running it: offer search, instance creation, SSH attach and tunnel, remote provisioning, model download, vLLM launch, an authenticated OpenCode session through the tunnel, and `coder stop` destroying the instance (confirmed gone in the Vast console).
>
> | Verified live | Not yet confirmed live |
> |---|---|
> | Offer search/filtering, balance/credit, instance list/lookup/create/destroy, SSH tunnel, model download, server launch, OpenCode integration, `coder stop` | The on-instance watchdog (deadline and idle shutdown), the S3 model cache, SGLang, models other than the one you tried, recovery paths (`coder cleanup`, Mac sleep/tunnel reconnect, failed-start cleanup on real hardware) |
> | Live testing found and fixed: URLs must end in `/` (the API 301-redirects slash-less URLs, unlike its docs); balance is `balance` + `credit`; CMP mining GPUs must be excluded | |
>
> The 76 unit tests use a mocked Vast API, a fake provider and fake SSH/remote layers. After every session, glance at the [Vast console](https://cloud.vast.ai/instances/) to confirm nothing is left running. See [Known limitations](#known-limitations).

## Install

Requires macOS/Linux, Python 3.12+, and the `ssh`/`ssh-keygen` binaries.

```bash
python3.13 -m venv .venv && .venv/bin/pip install -e '.[dev]'
ln -s "$PWD/.venv/bin/coder" /usr/local/bin/coder    # or add .venv/bin to PATH
```

## Credentials

| Secret | How | Required |
|---|---|---|
| Vast.ai API key ([create](https://cloud.vast.ai/manage-keys/)) | `export VAST_API_KEY=...` **or** `coder config --store-key vast` (macOS Keychain; `security` prompts, so the key never appears in argv) | yes |
| Hugging Face token | `HF_TOKEN` env or `coder config --store-key hf` | only for gated repos / faster downloads |
| S3 credentials | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` env | only with `storage.enabled` |

**`.env` files:** `coder` also reads `VAST_API_KEY`, `HF_TOKEN` and the `AWS_*` keys from `$CODER_ENV_FILE`, `./.env`, then `~/.config/coder/.env` (see [.env.example](.env.example)). Real environment variables win; only those allow-listed keys are imported, so an unrelated project `.env` can't inject anything else. `.env` and `.env.*` are git-ignored, and `.env` is never uploaded to the instance.

Keys are never read from or written to the config file (the config schema has no secret fields and rejects unknown keys).

## Configuration

`coder config --init` writes `~/.config/coder/config.yaml`, or copy [config.example.yaml](config.example.yaml). `coder config` prints the effective config. Errors are specific (`limits.max_hourly_price: Input should be greater than 0`). Custom models go in `~/.config/coder/models.yaml` (same schema as [models.yaml](src/coder/data/models.yaml); same key overrides built-in).

## Commands

| Command | What it does |
|---|---|
| `coder start [-m model] [--variant fp8\|bf16] [--engine vllm\|sglang] [--context N] [-y]` | Full startup. Reuses the active session (re-opens a dead tunnel) instead of renting a second box. |
| `coder stop [-y]` | Checks in-flight requests (warns), closes tunnel, restores OpenCode config, **destroys** the instance, verifies it is gone, prints cost. |
| `coder destroy [-y]` | Same, skipping graceful checks. Also the recovery path after a failed cleanup. |
| `coder status` | Provider, instance, GPU, VRAM, model, instance/server/tunnel state, duration, cost, month-to-date, warnings. |
| `coder offers [-m model]` | Compatible offers, best value first, with startup/session cost estimates. |
| `coder models [--details]` | Registry with requirements, repos, parsers, notes. |
| `coder logs [-n 100] [-f]` | Inference server log from the instance. |
| `coder cleanup [--destroy]` | Reconcile: list instances labelled `coder-managed:*` that no local session owns; optionally destroy them. |
| `coder restore-opencode` | Manually restore your previous OpenCode provider/model settings. |
| `coder config [--init] [--store-key vast\|hf]` | Show / set up configuration. |

**Stop vs. destroy.** The provider interface has `stop_instance` (compute stops, disk and its charges stay; Vast's docs state stopped instances keep billing storage), but no CLI command uses it. `coder stop` always destroys.

## Architecture

```
cli.py ─▶ lifecycle.Manager ─┬─ providers/{base,vast}.py   Provider ABC + Vast.ai REST client (retry/backoff on 429/5xx)
                             ├─ offers.py, costs.py        filter → rank → estimate; spend ledger
                             ├─ models.py + data/models.yaml   registry (pydantic-validated, user-extendable)
                             ├─ ssh.py                     ssh exec + tunnel via system ssh, dedicated key, per-session known_hosts
                             ├─ remote.py, inference.py    provision, resumable download, vLLM/SGLang command, health, watchdog
                             ├─ opencode.py                surgical merge into opencode.json + exact restore
                             └─ session.py                 0600 JSON state + flock; survives CLI crashes
```

A new provider (e.g. RunPod) means implementing `providers.base.Provider` and adding one branch to `providers/__init__.py`; the CLI, lifecycle and model code never import Vast. Everything is synchronous: each step depends on the previous one, so asyncio would add complexity without a speed-up.

### What happens in `coder start`

1. Validate config, credentials, model/variant/engine/context.
2. Search offers (server-side pre-filter, then an authoritative client-side filter on GPU count, per-GPU and total VRAM, price, reliability, system RAM, disk, bandwidth, country). Rank by price ÷ reliability², with a small penalty for slow downloads and a 10% bonus for preferred GPUs.
3. Show GPU, VRAM, $/h, startup estimate, expected session cost; refuse if the estimate exceeds `max_session_cost`; warn if it would blow the monthly budget; ask for confirmation (`-y` skips).
4. Create the instance (image `vllm/vllm-openai` or `lmsysorg/sglang`, `runtype: ssh_direct`, label `coder-managed:<id>`), falling back to the next offer if it was just rented. The session is **persisted immediately**.
5. Wait for `running`, attach the dedicated SSH public key (`POST /instances/{id}/ssh`), wait for SSH.
6. Write secrets to a `0600` file over ssh stdin; check GPU count and engine presence; start the self-destruct watchdog.
7. Download the model (detached on the host, polled, resumed up to 3 times on failure).
8. Launch the server bound to `127.0.0.1:8000` with `--api-key`, open `ssh -L 127.0.0.1:18000:127.0.0.1:8000`, poll the authenticated `/v1/models` through the tunnel until the model is listed (and detect a dead server process early, printing its log tail).
9. Write the OpenCode provider and print the ready message.

**Any** failure, timeout or Ctrl-C after step 4 triggers cleanup: close tunnel, restore OpenCode, destroy with retries, verify gone. If destruction cannot be verified, the session stays on disk as `destroy_failed`, a loud warning is printed, and `coder destroy` retries.

## Models and hardware

`coder models --details` shows everything below. Sizes and VRAM are my estimates from parameter count and checkpoint precision (Hugging Face cards don't publish disk size); the engines, versions, TP sizes and parser flags come from the model cards where stated, and `notes:` in the YAML flags where they are not.

| Key | Default checkpoint | Weights | Minimum rental | Notes |
|---|---|---|---|---|
| `qwen3-coder-next` | `Qwen/Qwen3-Coder-Next-FP8` | ~85 GB | **2× ≥40 GB GPUs (≥150 GB total)**, e.g. 2×H100/A100 80GB | 80B MoE. Cards require vLLM ≥0.15.0 / SGLang ≥0.5.8 and use TP=2. BF16 is ~160 GB, so a single 80 GB GPU **cannot** serve it (BF16 variant needs 4+ GPUs). |
| `qwen3-coder-30b-a3b` | `…-Instruct-FP8` | ~32 GB | 1× ≥40 GB | Cheapest option. |
| `devstral-small-2` | `mistralai/Devstral-Small-2-24B-Instruct-2512` (FP8) | ~26 GB | 1× ≥24 GB, ≥40 GB total | Card shows TP=2; single GPU is expected to work, undocumented. |
| `qwen3.8-27b` | `Qwen/Qwen3.8-27B` (BF16) | ~56 GB | ≥80 GB total | Card gives no serve flags; parsers copied from the Qwen3.6-27B card. |
| `gpt-oss-120b` | `openai/gpt-oss-120b` (MXFP4) | ~65 GB | 1× 80 GB | Parser `openai` is not stated on the card. |

GGUF is deliberately not offered: neither vLLM nor SGLang is a supported GGUF serving path for these checkpoints in the cards I read.

## Model download strategies

- **A, direct (default):** `huggingface_hub.snapshot_download` on the instance. Needs nothing from you.
- **B, S3-compatible cache (`storage.enabled`):** if `s3://bucket/prefix/<repo>/.coder-manifest.json` exists, `aws s3 sync` pulls only changed files and a size manifest is verified; otherwise the model is fetched from HF and published to the bucket for next time. `aws s3 sync` is never run with `--delete` in either direction and `coder` never deletes bucket objects. Integrity is **file-size based**, not hashing (hashing 80 GB costs several billed minutes). Best with a provider-adjacent, egress-free bucket (e.g. Cloudflare R2); AWS egress fees can exceed the saved download time.
- **Not implemented, by design: uploading from the Mac.** An 85 GB checkpoint at a typical 20–50 Mbps home upload is 4–10 hours, billed the whole time, versus roughly 5–15 minutes from HF on a datacenter link. It only pays off on a fast symmetric connection.

## OpenCode integration

`coder start` edits `~/.config/opencode/opencode.json` (override with `opencode.config_path`; VS Code is untouched). It writes only `provider.coder` (`@ai-sdk/openai-compatible`, `baseURL http://127.0.0.1:18000/v1`, the model with `limit.context`/`limit.output` and `tool_call`) and, if `set_default_model`, the top-level `model`. The keys written were checked against OpenCode's published JSON schema (`https://opencode.ai/config.json`, which rejects unknown keys). The prior values of those two keys are saved in `~/.local/state/coder/opencode-backup.json`; `coder stop` or `coder restore-opencode` restores them exactly, and leaves `model` alone if you switched it meanwhile. A `.jsonc` or invalid config is never modified (an error explains why). The per-session bearer token is written into this file (mode `0600`); it is worthless after the instance is destroyed and only valid through your localhost tunnel anyway.

## Cost controls

| Control | Enforcement |
|---|---|
| `max_hourly_price` | hard offer filter |
| `max_session_cost` | start refused if the estimate exceeds it; remote deadline = min(`max_session_hours`, `max_session_cost`/$h); `status` warns at `warn_at_fraction` |
| `max_session_hours` | **remote watchdog destroys the instance** at the deadline, independent of your Mac |
| `startup_timeout_minutes` | covers boot+download+load; failure ⇒ automatic destroy |
| `monthly_budget` | local ledger of finished sessions; warning/confirmation at start, shown in `status` |
| idle timeout | opt-in, see below |

Estimates use the offer's `dph_total` × elapsed time; they are not Vast's invoice. `coder` does not read your actual charges.

### Cost examples (illustrative; prices vary by the minute, check `coder offers`)

| Setup | $/h | Startup (est.) | 3 h session | 22 days/month |
|---|---|---|---|---|
| Qwen3-Coder-Next FP8 on 2×H100 | 3.00 | ~19 min ≈ $0.95 | ≈ $9.95 | ≈ $219 |
| Qwen3-Coder-Next FP8 on 2×A100 80GB | 1.60 | ~25 min ≈ $0.65 | ≈ $5.45 | ≈ $120 |
| Qwen3-Coder-30B-A3B FP8 on 1×A100 80GB | 0.80 | ~10 min ≈ $0.13 | ≈ $2.53 | ≈ $56 |

**At 2–4 h/day, $50–100/month rules out 2×H100 for daily use.** The 30B-A3B or gpt-oss-120b (1×80GB) fit the budget; Qwen3-Coder-Next fits at roughly 2×A100 prices for about 2 h/day. Every session pays startup minutes at full price, so fewer, longer sessions are cheaper than many short ones. (Table values are arithmetic from assumed prices, not quotes.)

## Idle shutdown (opt-in) and its limits

`idle.enabled: true` makes the on-instance watchdog destroy the instance after `timeout_minutes` (default 20) with no activity. It does not rely on your Mac. It counts completed `POST /v1/…` lines in the server log and reads the engine's Prometheus gauges (`vllm:num_requests_running/waiting`, `sglang:num_running_reqs/queue_reqs`) so a request in flight is never treated as idle. Limits: if those gauge names change, in-flight tracking silently disappears and a single streaming request longer than the idle timeout would be missed; SGLang needs `--enable-metrics` (we pass it). That is why it defaults to **off**. The deadline-based shutdown has no such dependency and is always on when Vast returns an instance-scoped key.

## Security notes

- The inference server binds `127.0.0.1` on the instance and requires a random per-session bearer token; the only way in is your SSH tunnel (bound to `127.0.0.1` locally). vLLM's `--api-key` covers `/v1` only; `/metrics` is open but also loopback-only.
- Secrets (token, HF/AWS credentials, provider key) travel over ssh stdin into `0600` files on the instance, never in `ssh` arguments. Inside the box, the API token is expanded by the remote shell, so it appears in the remote process list (the host admin can see it anyway).
- A dedicated ed25519 key (`~/.local/state/coder/ssh/`) is generated and used with `IdentitiesOnly`/`IdentityAgent=none`; your own keys, agent, `.env`, git credentials and project files are never uploaded.
- **Host keys: trust-on-first-use.** Vast's API does not expose host-key fingerprints, so `StrictHostKeyChecking=accept-new` pins the key on first connect into a per-session `known_hosts`; later connections in that session are checked. The first connection could be intercepted by a network attacker.
- **The cloud host can see everything the model sees**: every prompt, file excerpt and tool result you send, plus model weights and the instance's disk. Vast marketplace hosts are third parties; don't use it for code you can't expose. Datacenter-tier offers reduce but do not remove this.
- The instance-scoped Vast key lives on the instance (so the watchdog can self-destroy it). It can only act on that instance, per Vast's docs, and dies with it.
- Agent execution stays on your Mac; the model returns tool *calls* that OpenCode applies under its own permission settings. `coder` never executes anything the model emits.

## Reliability and recovery

State lives in `~/.local/state/coder/session.json` (mode `0600`, flock-guarded). Closed terminal or crash: `coder status` / `coder stop` still work. Mac sleep/network drop: the tunnel dies, the instance keeps running (and billing); `coder start` re-opens it. If you never come back, the watchdog's deadline destroys the box. Orphans (state lost, crash between create and save): `coder cleanup` lists `coder-managed:*` instances; `--destroy` removes only those, and instances without that label are never touched (checked again right before each delete). Provider calls retry with exponential backoff on 429/5xx/network errors (Vast sends no `Retry-After`).

## Tests

```bash
.venv/bin/pytest                                            # 76 tests, offline, nothing billable
CODER_LIVE_TESTS=1 VAST_API_KEY=… pytest -m live            # read-only API checks
CODER_LIVE_TESTS=1 CODER_LIVE_BILLABLE=1 VAST_API_KEY=… pytest -m live   # also rents the cheapest box for seconds
```

## Known limitations

- Not yet confirmed live: the remote watchdog, idle shutdown, S3 caching, SGLang, and the recovery paths on real hardware (see the top of this file). A full start → OpenCode → stop cycle has run on real Vast hardware.
- Vast doc inconsistencies handled defensively: endpoints need a trailing `/` (confirmed live; redirects are treated as errors); search `type` spelling (we retry `ondemand` after a 400); `gpu_name` formats (GPU names are matched client-side, not in the query); `inet_down` units are assumed Mb/s (the schema says MB/s, which would make estimates ~8× pessimistic); the account has both `balance` and `credit`, and we sum them (confirmed live).
- Not verified against official docs: the `/metrics` gauge names for in-flight tracking (from memory of vLLM/SGLang), tool-parser values for Qwen3-Coder-30B, gpt-oss-120b and Qwen3.8-27B, and the instance-scoped key's permission to `DELETE` (stated in Vast's hello-world guide; if it fails, the watchdog logs it in `/root/coder/watchdog.log`).
- Docker image tags default to `latest`; pin tags in `images:` once you find a working combination. Qwen3-Coder-Next needs vLLM ≥0.15.0.
- Vast's `onstart` is not used; everything runs over SSH after boot, so a few extra minutes of billed time are possible versus a pre-baked image.
- Destruction verification relies on `GET /instances/{id}` returning 404/empty after `DELETE`; if Vast keeps a "destroyed" row visible, `coder` reports `destroy_failed` conservatively. Confirm in the [Vast console](https://cloud.vast.ai/instances/) after your first session.
- One session at a time; one local tunnel port.
