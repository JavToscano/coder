"""Non-destructive OpenCode configuration management.

We only ever touch two things in the user's opencode.json: `provider.<provider_id>` and
(optionally) the top-level `model`. Their previous values are recorded in a backup file so
`restore` can put back exactly what was there, leaving everything else byte-for-byte equivalent
in content (key order is preserved; JSON formatting is normalised).
Schema reference: https://opencode.ai/config.json (additionalProperties is false, so only
documented model keys are written).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from . import paths
from .errors import OpenCodeConfigError

SCHEMA_URL = "https://opencode.ai/config.json"


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        jsonc = path.with_suffix(".jsonc")
        if jsonc.exists():
            raise OpenCodeConfigError(
                f"Found {jsonc} but not {path}. coder cannot rewrite JSONC without losing comments. "
                f"Set opencode.config_path in the coder config to a JSON file, or convert your config to opencode.json."
            )
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise OpenCodeConfigError(f"{path} is not valid JSON ({e}); refusing to modify it.") from e
    if not isinstance(data, dict):
        raise OpenCodeConfigError(f"{path}: top level must be an object")
    return data


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".coder-tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # contains the session token
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def provider_block(
    *, display_name: str, model_id: str, base_url: str, api_key: str,
    context: int, output: int, tool_call: bool,
) -> dict[str, Any]:
    return {
        "npm": "@ai-sdk/openai-compatible",
        "name": "coder (ephemeral cloud GPU)",
        "options": {"baseURL": base_url, "apiKey": api_key},
        "models": {
            model_id: {
                "name": display_name,
                "tool_call": tool_call,
                "limit": {"context": context, "output": output},
            }
        },
    }


def apply(
    path: Path, provider_id: str, block: dict[str, Any], model_id: str, *, set_default_model: bool,
    backup_path: Path | None = None,
) -> str:
    """Install/refresh the provider. Returns the `provider/model` id to select in OpenCode."""
    backup_path = backup_path or paths.opencode_backup_file()
    data = _read(path)
    full_model = f"{provider_id}/{model_id}"

    existing = _load_backup(backup_path)
    if existing is None or existing.get("path") != str(path):
        providers = data.get("provider") if isinstance(data.get("provider"), dict) else {}
        existing = {
            "path": str(path),
            "provider_id": provider_id,
            "file_existed": path.exists(),
            "had_schema": "$schema" in data,
            "had_provider_key": "provider" in data,
            "had_provider": provider_id in providers,
            "previous_provider": providers.get(provider_id),
            "had_model": "model" in data,
            "previous_model": data.get("model"),
            "set_model": None,
        }
    # (If a backup for this path already exists we keep it: it holds the user's true originals,
    # not our own previous write.)
    data.setdefault("$schema", SCHEMA_URL)
    if not isinstance(data.get("provider"), dict):
        data["provider"] = {}
    data["provider"][provider_id] = block
    if set_default_model:
        data["model"] = full_model
        existing["set_model"] = full_model
    _save_backup(backup_path, existing)
    _write(path, data)
    return full_model


def restore(backup_path: Path | None = None) -> bool:
    """Undo `apply`. Returns False if there was nothing to restore."""
    backup_path = backup_path or paths.opencode_backup_file()
    b = _load_backup(backup_path)
    if b is None:
        return False
    path = Path(b["path"])
    if path.exists():
        data = _read(path)
        providers = data.get("provider") if isinstance(data.get("provider"), dict) else {}
        pid = b["provider_id"]
        if b["had_provider"]:
            providers[pid] = b["previous_provider"]
        else:
            providers.pop(pid, None)
        if providers or b["had_provider_key"]:
            data["provider"] = providers
        else:
            data.pop("provider", None)
        # Only touch `model` if it still points at ours; the user may have switched since.
        if b.get("set_model") and data.get("model") == b["set_model"]:
            if b["had_model"]:
                data["model"] = b["previous_model"]
            else:
                data.pop("model", None)
        if not b.get("had_schema", True):
            data.pop("$schema", None)
        if not b["file_existed"] and not data:
            path.unlink()
        else:
            _write(path, data)
    backup_path.unlink(missing_ok=True)
    return True


def is_applied(backup_path: Path | None = None) -> bool:
    return _load_backup(backup_path or paths.opencode_backup_file()) is not None


def _load_backup(p: Path) -> dict[str, Any] | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _save_backup(p: Path, b: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(b, f, indent=2)
