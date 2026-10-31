"""Vast.ai provider.

Endpoints used (docs.vast.ai, host https://console.vast.ai). IMPORTANT: the live API answers
every URL WITHOUT a trailing slash with an HTTP 301 to the slashed URL (the docs omit the
slashes), so every path below carries one:
  POST   /api/v0/bundles/                 search offers
  PUT    /api/v0/asks/{offer_id}/         create instance from an offer
  GET    /api/v0/instances/{id}/          one instance  -> {"instances": {...}}
  GET    /api/v1/instances/               list (keyset pagination via next_token)
  PUT    /api/v0/instances/{id}/          {"state": "running"|"stopped"}
  DELETE /api/v0/instances/{id}/          destroy
  POST   /api/v0/instances/{id}/ssh/      {"ssh_key": "..."}
  GET    /api/v0/users/current/           balance
Rate limiting: HTTP 429 without Retry-After, so we back off ourselves.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

import httpx

from ..errors import OfferUnavailable, ProviderError
from .base import (
    MANAGED_LABEL_PREFIX, Billing, CreatedInstance, Instance, InstanceSpec, InstanceState,
    Offer, OfferQuery, Provider, SSHInfo,
)

log = logging.getLogger(__name__)
BASE_URL = "https://console.vast.ai"
MB_PER_GB = 1000  # Vast reports RAM in MB; its own CLI converts with 1000


class VastProvider(Provider):
    name = "vast"

    def __init__(
        self,
        api_key: str,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 5,
    ) -> None:
        self._client = httpx.Client(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            timeout=httpx.Timeout(30.0),
            transport=transport,
        )
        self._sleep = sleep
        self._max_retries = max_retries

    def close(self) -> None:
        self._client.close()

    # -- http ---------------------------------------------------------------
    def _request(self, method: str, url: str, *, ok_404: bool = False, **kw: Any) -> dict[str, Any] | None:
        delay = 1.5
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.request(method, url, **kw)
            except httpx.TransportError as e:
                if attempt == self._max_retries:
                    raise ProviderError(f"Vast.ai unreachable ({method} {url}): {e}") from e
                self._sleep(delay); delay = min(delay * 2, 30)
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == self._max_retries:
                    raise ProviderError(f"Vast.ai {method} {url} failed after retries: HTTP {resp.status_code} {resp.text[:200]}", resp.status_code)
                self._sleep(delay); delay = min(delay * 2, 30)
                continue
            if resp.status_code in (301, 302, 307, 308):
                # Never silently treat a redirect as success (this once hid a failed DELETE).
                raise ProviderError(f"Vast.ai {method} {url} was redirected to {resp.headers.get('location')}; refusing to guess.", resp.status_code)
            if resp.status_code == 404 and ok_404:
                return None
            if resp.status_code in (401, 403):
                raise ProviderError("Vast.ai rejected the API key (HTTP %d). Check VAST_API_KEY." % resp.status_code, resp.status_code)
            if resp.status_code >= 400:
                raise ProviderError(f"Vast.ai {method} {url}: HTTP {resp.status_code} {resp.text[:300]}", resp.status_code)
            try:
                body = resp.json()
            except json.JSONDecodeError:
                raise ProviderError(f"Vast.ai {method} {url}: non-JSON response (HTTP {resp.status_code}): {resp.text[:120]!r}", resp.status_code) from None
            if isinstance(body, dict) and body.get("success") is False:
                raise ProviderError(f"Vast.ai {method} {url}: {body.get('error')}: {body.get('msg')}", resp.status_code)
            return body if isinstance(body, dict) else {"_": body}
        raise AssertionError("unreachable")

    # -- offers -------------------------------------------------------------
    def search_offers(self, query: OfferQuery) -> list[Offer]:
        body: dict[str, Any] = {
            "verified": {"eq": True},
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "num_gpus": {"in": sorted(set(query.gpu_counts))},
            "gpu_ram": {"gte": query.min_gpu_vram_gb * MB_PER_GB},  # per-GPU, MB
            "cpu_ram": {"gte": query.min_cpu_ram_gb * MB_PER_GB},
            "disk_space": {"gte": query.min_disk_gb},
            "reliability": {"gte": query.min_reliability},
            "dph_total": {"lte": query.max_hourly_price},
            "direct_port_count": {"gte": 1},  # needed for ssh_direct
            "order": [["dph_total", "asc"]],
            "limit": query.limit,
        }
        # Docs disagree on the enum spelling ("on-demand" in guides, "ondemand" in the schema).
        for type_value in ("on-demand", "ondemand"):
            try:
                resp = self._request("POST", "/api/v0/bundles/", json={**body, "type": type_value})
                break
            except ProviderError as e:
                if e.status != 400 or type_value == "ondemand":
                    raise
        else:  # pragma: no cover
            raise AssertionError
        return [self._offer(o) for o in (resp or {}).get("offers", [])]

    @staticmethod
    def _offer(o: dict[str, Any]) -> Offer:
        return Offer(
            provider="vast",
            id=int(o["id"]),
            gpu_name=str(o.get("gpu_name", "")),
            num_gpus=int(o.get("num_gpus", 1)),
            gpu_ram_gb=float(o.get("gpu_ram", 0)) / MB_PER_GB,
            cpu_ram_gb=float(o.get("cpu_ram", 0)) / MB_PER_GB,
            disk_gb=float(o.get("disk_space", 0)),
            dph_total=float(o.get("dph_total", 0)),
            reliability=float(o.get("reliability", o.get("reliability2", 0)) or 0),
            geolocation=str(o.get("geolocation", "")),
            inet_down_mbps=float(o.get("inet_down", 0) or 0),  # Vast reports Mb/s in practice
            raw=o,
        )

    # -- instances ----------------------------------------------------------
    def create_instance(self, offer: Offer, spec: InstanceSpec) -> CreatedInstance:
        body = {
            "image": spec.image,
            "label": spec.label,
            "disk": spec.disk_gb,
            "runtype": "ssh_direct",  # sshd on a directly reachable port, no proxy hop
        }
        try:
            resp = self._request("PUT", f"/api/v0/asks/{offer.id}/", json=body) or {}
        except ProviderError as e:
            if e.status in (404, 410):
                raise OfferUnavailable(f"Offer {offer.id} is no longer available", e.status) from e
            raise
        if "new_contract" not in resp:
            raise ProviderError(f"Vast.ai create returned no instance id: {resp}")
        return CreatedInstance(id=int(resp["new_contract"]), instance_api_key=resp.get("instance_api_key"))

    def get_instance(self, instance_id: int) -> Instance | None:
        resp = self._request("GET", f"/api/v0/instances/{instance_id}/", ok_404=True)
        if resp is None:
            return None
        if "instances" not in resp:
            raise ProviderError(f"Vast.ai get instance {instance_id}: unexpected response {str(resp)[:200]}")
        inst = resp["instances"]  # null once the instance is gone
        if isinstance(inst, list):
            inst = inst[0] if inst else None
        if not inst:
            return None
        return self._instance(inst)

    def list_managed_instances(self) -> list[Instance]:
        out: list[Instance] = []
        token: str | None = None
        while True:
            params: dict[str, Any] = {"limit": 25}
            if token:
                params["after_token"] = token
            resp = self._request("GET", "/api/v1/instances/", params=params) or {}
            for raw in resp.get("instances", []):
                if str(raw.get("label") or "").startswith(MANAGED_LABEL_PREFIX):
                    out.append(self._instance(raw))
            token = resp.get("next_token")
            if not token:
                return out

    @staticmethod
    def _instance(r: dict[str, Any]) -> Instance:
        actual = r.get("actual_status")
        cur = r.get("cur_state")
        intended = r.get("intended_status")
        if actual == "running":
            state = InstanceState.RUNNING
        elif cur == "stopped" or intended == "stopped" or actual == "stopped":
            state = InstanceState.STOPPED
        elif actual in (None, "loading", "created", "creating") or (actual is None and cur == "running"):
            state = InstanceState.PENDING
        elif actual in ("exited", "offline", "unknown"):
            state = InstanceState.FAILED
        else:
            state = InstanceState.PENDING
        ssh = None
        ports = r.get("ports")
        host = (r.get("public_ipaddr") or "").strip()
        if isinstance(ports, dict) and host and ports.get("22/tcp"):
            try:
                ssh = SSHInfo(host, int(ports["22/tcp"][0]["HostPort"]))
            except (KeyError, IndexError, ValueError, TypeError):
                ssh = None
        if ssh is None and r.get("ssh_host") and r.get("ssh_port"):
            ssh = SSHInfo(str(r["ssh_host"]), int(r["ssh_port"]))
        return Instance(
            provider="vast",
            id=int(r["id"]),
            state=state,
            label=str(r.get("label") or ""),
            raw_status=str(actual or cur or ""),
            status_msg=str(r.get("status_msg") or ""),
            ssh=ssh,
            dph_total=float(r.get("dph_total") or 0),
            gpu_name=str(r.get("gpu_name") or ""),
            num_gpus=int(r.get("num_gpus") or 0),
            gpu_ram_gb=float(r.get("gpu_ram") or 0) / MB_PER_GB,
            start_epoch=float(r["start_date"]) if r.get("start_date") else None,
            raw=r,
        )

    def start_instance(self, instance_id: int) -> None:
        self._request("PUT", f"/api/v0/instances/{instance_id}/", json={"state": "running"})

    def stop_instance(self, instance_id: int) -> None:
        self._request("PUT", f"/api/v0/instances/{instance_id}/", json={"state": "stopped"})

    def destroy_instance(self, instance_id: int) -> None:
        self._request("DELETE", f"/api/v0/instances/{instance_id}/", ok_404=True)

    def attach_ssh_key(self, instance_id: int, public_key: str) -> None:
        self._request("POST", f"/api/v0/instances/{instance_id}/ssh/", json={"ssh_key": public_key.strip()})

    def get_billing(self) -> Billing | None:
        # The live response has BOTH `balance` (cash) and `credit` (prepaid/signup credit); the docs
        # mention only one or the other depending on the page. Spendable funds are their sum.
        resp = self._request("GET", "/api/v0/users/current/") or {}
        parts = [resp.get(k) for k in ("balance", "credit") if isinstance(resp.get(k), (int, float))]
        return Billing(balance_usd=float(sum(parts)) if parts else None)
