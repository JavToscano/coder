"""Filesystem locations. `CODER_HOME` relocates everything (used by tests)."""
from __future__ import annotations

import os
from pathlib import Path


def config_dir() -> Path:
    if home := os.environ.get("CODER_HOME"):
        return Path(home) / "config"
    return Path.home() / ".config" / "coder"


def state_dir() -> Path:
    if home := os.environ.get("CODER_HOME"):
        return Path(home) / "state"
    return Path.home() / ".local" / "state" / "coder"


def config_file() -> Path:
    return config_dir() / "config.yaml"


def user_models_file() -> Path:
    return config_dir() / "models.yaml"


def session_file() -> Path:
    return state_dir() / "session.json"


def ledger_file() -> Path:
    return state_dir() / "ledger.jsonl"


def opencode_backup_file() -> Path:
    return state_dir() / "opencode-backup.json"


def ssh_dir() -> Path:
    return state_dir() / "ssh"


def opencode_config_file() -> Path:
    if home := os.environ.get("CODER_HOME"):
        return Path(home) / "opencode" / "opencode.json"
    return Path.home() / ".config" / "opencode" / "opencode.json"
