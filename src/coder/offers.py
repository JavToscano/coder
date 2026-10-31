"""Offer query construction, client-side filtering and value ranking."""
from __future__ import annotations

import re

from .config import Config
from .models import ModelSpec, Variant
from .providers.base import Offer, OfferQuery

# Image pull + OS overhead on top of the checkpoint.
IMAGE_AND_SCRATCH_GB = 45


def required_disk_gb(variant: Variant) -> int:
    return int(variant.download_gb * 1.15 + IMAGE_AND_SCRATCH_GB)


def build_query(cfg: Config, variant: Variant) -> OfferQuery:
    return OfferQuery(
        gpu_counts=variant.gpus,
        min_gpu_vram_gb=max(variant.min_gpu_vram_gb, cfg.offers.min_vram_gb),
        min_total_vram_gb=variant.min_total_vram_gb,
        max_hourly_price=cfg.limits.max_hourly_price,
        min_reliability=cfg.offers.min_reliability,
        min_cpu_ram_gb=cfg.offers.min_system_ram_gb,
        min_disk_gb=required_disk_gb(variant),
        min_inet_down_mbps=cfg.offers.min_inet_down_mbps,
        countries=[c.upper() for c in cfg.offers.countries],
        exclude_gpus=cfg.offers.exclude_gpus,
    )


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _matches_preferred(offer: Offer, preferred: list[str]) -> bool:
    n = _norm(offer.gpu_name)
    return any(_norm(p) in n for p in preferred)


def filter_offers(offers: list[Offer], q: OfferQuery) -> list[Offer]:
    """Authoritative filter; the provider-side query is only a pre-filter."""
    wanted = [_norm(g) for g in q.gpu_names]
    banned = [_norm(g) for g in q.exclude_gpus]
    out = []
    for o in offers:
        if o.num_gpus not in q.gpu_counts:
            continue
        if o.gpu_ram_gb < q.min_gpu_vram_gb or o.total_vram_gb < q.min_total_vram_gb:
            continue
        if o.dph_total > q.max_hourly_price or o.dph_total <= 0:
            continue
        if o.reliability < q.min_reliability or o.cpu_ram_gb < q.min_cpu_ram_gb or o.disk_gb < q.min_disk_gb:
            continue
        if q.min_inet_down_mbps and o.inet_down_mbps < q.min_inet_down_mbps:
            continue
        if q.countries and o.country not in q.countries:
            continue
        if any(b in _norm(o.gpu_name) for b in banned):
            continue
        if wanted and not any(w in _norm(o.gpu_name) for w in wanted):
            continue
        out.append(o)
    return out


def value_score(o: Offer, preferred: list[str]) -> float:
    """Lower is better: effective $/hr, penalised for flaky hosts and slow downloads,
    with a 10% bonus for preferred GPU families (a tie-breaker, not an override)."""
    score = o.dph_total / max(o.reliability, 0.5) ** 2
    if o.inet_down_mbps:
        score *= 1 + min(0.15, 150 / o.inet_down_mbps * 0.1)
    if _matches_preferred(o, preferred):
        score *= 0.9
    return score


def rank_offers(offers: list[Offer], preferred: list[str]) -> list[Offer]:
    return sorted(offers, key=lambda o: (value_score(o, preferred), o.dph_total, -o.reliability, o.id))


def find_offers(provider, cfg: Config, model: ModelSpec, variant: Variant) -> list[Offer]:
    q = build_query(cfg, variant)
    return rank_offers(filter_offers(provider.search_offers(q), q), cfg.offers.preferred_gpus)
