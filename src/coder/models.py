"""Model registry: YAML-defined models with per-checkpoint serving requirements."""
from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from . import paths
from .errors import ConfigError

Engine = Literal["vllm", "sglang"]


class Variant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo: str
    quant: str
    download_gb: float = Field(gt=0)
    min_total_vram_gb: float = Field(gt=0)
    min_gpu_vram_gb: float = Field(gt=0)
    gpus: list[int] = Field(min_length=1)
    engines: list[Engine] = Field(min_length=1)
    min_engine_version: dict[str, str] = Field(default_factory=dict)
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    extra_args: dict[str, list[str]] = Field(default_factory=dict)  # per-engine extra CLI args
    notes: str = ""


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str = ""
    display_name: str
    served_name: str
    description: str = ""
    context_length: int = Field(gt=0)
    default_context: int | None = None
    tool_calling: bool = True
    sampling: dict[str, float] = Field(default_factory=dict)
    default_variant: str
    variants: dict[str, Variant]

    @model_validator(mode="after")
    def _check(self) -> "ModelSpec":
        if self.default_variant not in self.variants:
            raise ValueError(f"default_variant '{self.default_variant}' not in variants {list(self.variants)}")
        if self.tool_calling and not any(v.tool_call_parser for v in self.variants.values()):
            raise ValueError("tool_calling is true but no variant defines tool_call_parser")
        return self

    def variant(self, name: str | None) -> tuple[str, Variant]:
        name = name or self.default_variant
        if name not in self.variants:
            raise ConfigError(f"Model '{self.key}' has no variant '{name}'. Available: {', '.join(self.variants)}")
        return name, self.variants[name]

    def effective_context(self, override: int | None) -> int:
        ctx = override or self.default_context or self.context_length
        if ctx > self.context_length:
            raise ConfigError(f"Requested context {ctx} exceeds {self.display_name}'s maximum {self.context_length}")
        return ctx


class Registry:
    def __init__(self, models: dict[str, ModelSpec]) -> None:
        self._models = models

    def __iter__(self):
        return iter(self._models.values())

    def keys(self) -> list[str]:
        return list(self._models)

    def get(self, key: str) -> ModelSpec:
        try:
            return self._models[key]
        except KeyError:
            raise ConfigError(f"Unknown model '{key}'. Known models: {', '.join(self._models)}") from None


def _parse(raw: Any, source: str) -> dict[str, ModelSpec]:
    if not isinstance(raw, dict) or not isinstance(raw.get("models"), dict):
        raise ConfigError(f"{source}: expected a top-level 'models' mapping")
    out: dict[str, ModelSpec] = {}
    for key, body in raw["models"].items():
        try:
            out[key] = ModelSpec.model_validate({**body, "key": key})
        except ValidationError as e:
            msgs = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())
            raise ConfigError(f"{source}: model '{key}' is invalid: {msgs}") from e
    return out


def load_registry(user_file: Path | None = None) -> Registry:
    builtin = yaml.safe_load(resources.files("coder").joinpath("data/models.yaml").read_text())
    models = _parse(builtin, "built-in models.yaml")
    user_file = user_file or paths.user_models_file()
    if user_file.exists():
        try:
            models.update(_parse(yaml.safe_load(user_file.read_text()), str(user_file)))
        except yaml.YAMLError as e:
            raise ConfigError(f"{user_file}: invalid YAML: {e}") from e
    return Registry(models)
