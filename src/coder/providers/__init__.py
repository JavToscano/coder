from __future__ import annotations

from ..config import Config, vast_api_key
from ..errors import ConfigError
from .base import Provider


def build_provider(cfg: Config) -> Provider:
    """Provider factory. Adding RunPod = implement `Provider` and register it here."""
    if cfg.provider == "vast":
        from .vast import VastProvider

        return VastProvider(vast_api_key())
    raise ConfigError(f"Unknown provider '{cfg.provider}'. Supported: vast")
