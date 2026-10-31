"""Provider-neutral types and the interface every cloud backend implements."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

MANAGED_LABEL_PREFIX = "coder-managed"


@dataclass(frozen=True)
class Offer:
    provider: str
    id: int
    gpu_name: str
    num_gpus: int
    gpu_ram_gb: float  # per GPU
    cpu_ram_gb: float
    disk_gb: float
    dph_total: float  # USD / hour for the whole machine, storage included
    reliability: float
    geolocation: str
    inet_down_mbps: float
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def total_vram_gb(self) -> float:
        return self.gpu_ram_gb * self.num_gpus

    @property
    def country(self) -> str:
        return self.geolocation.rsplit(",", 1)[-1].strip().upper()


@dataclass
class OfferQuery:
    gpu_counts: list[int]
    min_gpu_vram_gb: float
    min_total_vram_gb: float
    max_hourly_price: float
    min_reliability: float = 0.0
    min_cpu_ram_gb: float = 0.0
    min_disk_gb: float = 0.0
    min_inet_down_mbps: float = 0.0
    gpu_names: list[str] = field(default_factory=list)  # substring match, empty = any
    exclude_gpus: list[str] = field(default_factory=list)
    countries: list[str] = field(default_factory=list)
    limit: int = 64


class InstanceState(str, Enum):
    PENDING = "pending"  # created, image pulling / booting
    RUNNING = "running"
    STOPPED = "stopped"  # still exists and still bills for storage
    FAILED = "failed"  # exited / offline / unknown: will never become RUNNING
    GONE = "gone"


@dataclass(frozen=True)
class SSHInfo:
    host: str
    port: int


@dataclass
class Instance:
    provider: str
    id: int
    state: InstanceState
    label: str = ""
    raw_status: str = ""
    status_msg: str = ""
    ssh: SSHInfo | None = None
    dph_total: float = 0.0
    gpu_name: str = ""
    num_gpus: int = 0
    gpu_ram_gb: float = 0.0
    start_epoch: float | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class InstanceSpec:
    image: str
    label: str
    disk_gb: int


@dataclass(frozen=True)
class CreatedInstance:
    id: int
    # Restricted key scoped to this instance (if the provider issues one); lets the remote
    # watchdog destroy exactly this instance and nothing else.
    instance_api_key: str | None = None


@dataclass(frozen=True)
class Billing:
    balance_usd: float | None


class Provider(ABC):
    name: str

    @abstractmethod
    def search_offers(self, query: OfferQuery) -> list[Offer]: ...

    @abstractmethod
    def create_instance(self, offer: Offer, spec: InstanceSpec) -> CreatedInstance: ...

    @abstractmethod
    def get_instance(self, instance_id: int) -> Instance | None:
        """None when the instance no longer exists."""

    @abstractmethod
    def list_managed_instances(self) -> list[Instance]:
        """Instances whose label starts with MANAGED_LABEL_PREFIX. Never returns foreign ones."""

    @abstractmethod
    def start_instance(self, instance_id: int) -> None: ...

    @abstractmethod
    def stop_instance(self, instance_id: int) -> None:
        """Stops compute billing but KEEPS the instance and its disk (storage keeps billing)."""

    @abstractmethod
    def destroy_instance(self, instance_id: int) -> None:
        """Permanently deletes the instance and its disk."""

    @abstractmethod
    def attach_ssh_key(self, instance_id: int, public_key: str) -> None: ...

    @abstractmethod
    def get_billing(self) -> Billing | None: ...

    def close(self) -> None:  # pragma: no cover - optional
        pass
