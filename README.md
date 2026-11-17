# coder

Rent a cloud GPU on Vast.ai only while you code, serve a coding LLM on it, point OpenCode at it through an SSH tunnel, then **destroy everything** when you stop. Nothing persists, so there is no storage bill between sessions.

```
coder start --model qwen3-coder-next   # find offer → confirm → boot → download → serve → tunnel → OpenCode
opencode                               # use it from your Mac, project files stay local
coder stop                             # tunnel closed, OpenCode config restored, instance destroyed + verified
```

**Status:** a full start → OpenCode → stop session has run on real Vast.ai. Not yet confirmed live: the on-instance watchdog (deadline and idle shutdown), the S3 model cache, SGLang, other models, and the recovery paths. After every session, check the [Vast console](https://cloud.vast.ai/instances/) to confirm nothing is left running.

## Install

Requires macOS/Linux, Python 3.12+, and `ssh`/`ssh-keygen`.

```bash
python3.13 -m venv .venv && .venv/bin/pip install -e '.[dev]'
ln -s "$PWD/.venv/bin/coder" /usr/local/bin/coder    # or add .venv/bin to PATH
```

## Setup

**Credentials.** Keys are never stored in the config file.

| Secret | How | Required |
|---|---|---|
| Vast.ai API key ([create](https://cloud.vast.ai/manage-keys/)) | `export VAST_API_KEY=...` or `coder config --store-key vast` (macOS Keychain) | yes |
| Hugging Face token | `HF_TOKEN` or `coder config --store-key hf` | gated repos / faster downloads |
| S3 credentials | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | only with `storage.enabled` |

`coder` also reads those keys (and only those) from `$CODER_ENV_FILE`, `./.env`, then `~/.config/coder/.env`; see [.env.example](.env.example). Real environment variables win, and `.env` is never uploaded.

**Config.** `coder config --init` writes `~/.config/coder/config.yaml` (or copy [config.example.yaml](config.example.yaml)); `coder config` prints the effective config. Custom models go in `~/.config/coder/models.yaml`, same schema as [models.yaml](src/coder/data/models.yaml).

## Commands

| Command | What it does |
|---|---|
| `coder start [-m model] [--variant fp8\|bf16] [--engine vllm\|sglang] [--context N] [-o N] [-y]` | Full startup. Reuses an active session instead of renting a second box. In a terminal, arrow-key menus (↑/↓, Enter, `q` to cancel) pick the model (skipped with `-m`) and the offer (top 8; skipped with `-o N`). `-y` skips both and rents the best offer. |
| `coder stop [-y]` | Warns about in-flight requests, closes the tunnel, restores OpenCode config, destroys the instance, verifies it is gone, prints cost. |
| `coder destroy [-y]` | Same, skipping graceful checks. Also the recovery path after a failed cleanup. |
| `coder status` | Instance, GPU, model, tunnel state, duration, cost, month-to-date, warnings. |
| `coder offers [-m model] [--top N]` | Compatible offers, best value first, with cost estimates. `start -o N` rents row N (no fallback hosts). |
| `coder models [--details]` | Model registry with requirements and notes. |
| `coder logs [-n 100] [-f]` | Inference server log from the instance. |
| `coder cleanup [--destroy]` | Find (and optionally destroy) `coder-managed:*` instances no local session owns. |
| `coder restore-opencode` | Manually restore your previous OpenCode settings. |
| `coder config [--init] [--store-key vast\|hf]` | Show / set up configuration. |

`coder stop` always destroys; stopped Vast instances keep billing for disk.

## Models

Sizes and VRAM are estimates from parameter count and precision. Run `coder models --details` for the full list.

| Key | Weights | Minimum rental |
|---|---|---|
| `qwen3-coder-next` (FP8) | ~85 GB | 2× ≥40 GB GPUs (≥150 GB total), e.g. 2×A100/H100 80GB |
| `qwen3-coder-next-abliterated` (BF16) | ~160 GB | 4× ≥48 GB |
| `qwen3-coder-30b-a3b` (FP8) | ~32 GB | 1× ≥40 GB (cheapest) |
| `qwen3-coder-30b-a3b-abliterated` (BF16) | ~62 GB | 1× ≥40 GB |
| `devstral-small-2` (FP8) | ~26 GB | 1× ≥24 GB |
| `qwen3.8-27b` (BF16) | ~56 GB | ≥80 GB total |
| `gpt-oss-120b` (MXFP4) | ~65 GB | 1× 80 GB |

Models are downloaded on the instance from Hugging Face. Optionally (`storage.enabled`), an S3-compatible bucket caches weights between sessions; it never deletes bucket objects and checks integrity by file size only. Uploading from your Mac is not supported: at home upload speeds it takes hours of billed time.

## Cost controls

| Setting | Effect |
|---|---|
| `max_hourly_price` | Hard offer filter. |
| `max_session_cost` | Start is refused if the estimate exceeds it; also sets the remote deadline. |
| `max_session_hours` | A watchdog on the instance destroys it at the deadline, independent of your Mac. |
| `startup_timeout_minutes` | Boot + download + load must finish in time, or the instance is destroyed. |
| `monthly_budget` | Local ledger; warns at start and in `status`. |
| `idle.enabled` | Opt-in: destroy after `timeout_minutes` (default 20) with no requests. Off by default because it depends on engine metric names. |

Estimates are the offer's hourly price × elapsed time, not Vast's invoice. Every session pays startup minutes at full price, so fewer, longer sessions are cheaper. Rough guide: 2×A100 80GB is about $1.60/h, 1×A100 about $0.80/h; check `coder offers` for live prices.

## OpenCode integration

`coder start` edits `~/.config/opencode/opencode.json` (override with `opencode.config_path`). It writes only `provider.coder` (pointing at `http://127.0.0.1:18000/v1`) and, if `set_default_model`, the top-level `model`. Previous values are backed up and restored exactly by `coder stop` or `coder restore-opencode`. A `.jsonc` or invalid config is never modified.

## Security

- The server binds `127.0.0.1` on the instance with a random per-session bearer token; the only way in is your SSH tunnel.
- Secrets travel over ssh stdin into `0600` files, never in command arguments. A dedicated SSH key is used; your own keys, agent, `.env`, and project files are never uploaded.
- Host keys are trust-on-first-use (Vast doesn't expose fingerprints).
- **The cloud host can see everything the model sees**: prompts, file excerpts, tool results. Don't use it for code you can't expose.
- Agent execution stays on your Mac; `coder` never runs anything the model emits.

## Recovery

State lives in `~/.local/state/coder/session.json`. After a crash or closed terminal, `coder status` / `coder stop` still work. If your Mac sleeps, the tunnel dies but the instance keeps billing; `coder start` re-opens it, or the watchdog destroys it at the deadline. Any failure during `start` triggers automatic cleanup; if destruction can't be verified, `coder destroy` retries. `coder cleanup` finds orphans and only ever touches instances labelled `coder-managed:*`.

## Architecture

```
cli.py ─▶ lifecycle.Manager ─┬─ providers/{base,vast}.py       Provider ABC + Vast.ai client
                             ├─ offers.py, costs.py            filter → rank → estimate; spend ledger
                             ├─ models.py + data/models.yaml   model registry
                             ├─ ssh.py                         exec + tunnel via system ssh
                             ├─ remote.py, inference.py        provision, download, vLLM/SGLang, watchdog
                             ├─ opencode.py                    config merge + exact restore
                             └─ session.py                     0600 JSON state + flock
```

A new provider means implementing `providers.base.Provider` and adding one branch in `providers/__init__.py`.

## Tests

```bash
.venv/bin/pytest                                            # offline, nothing billable
CODER_LIVE_TESTS=1 VAST_API_KEY=… pytest -m live            # read-only API checks
CODER_LIVE_TESTS=1 CODER_LIVE_BILLABLE=1 VAST_API_KEY=… pytest -m live   # also rents the cheapest box for seconds
```

## Known limitations

- One session at a time; one local tunnel port.
- Docker image tags default to `latest`; pin them in `images:` once you find a working combination. Qwen3-Coder-Next needs vLLM ≥0.15.0.
- Vast API quirks are handled defensively (trailing `/` on URLs, `inet_down` units assumed Mb/s, `balance` + `credit` summed).
- Unverified against official docs: engine `/metrics` gauge names, some tool-parser values (Qwen3-Coder-30B, gpt-oss-120b, Qwen3.8-27B), and whether the instance-scoped key can `DELETE` itself (if not, see `/root/coder/watchdog.log`).
- Destruction is verified via `GET /instances/{id}` returning 404/empty; if Vast keeps a "destroyed" row visible, `coder` reports `destroy_failed` conservatively. Confirm in the Vast console.
