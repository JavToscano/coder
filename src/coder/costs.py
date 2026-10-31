"""Cost estimation and the local spend ledger."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from . import paths
from .providers.base import Offer

PROVISION_MIN = 4.0  # create -> ssh reachable, typical
LOAD_MIN_PER_100GB = 4.0  # weights into VRAM + CUDA graph capture
MIN_LOAD_MIN = 2.0
DOWNLOAD_EFFICIENCY = 0.5  # real-world fraction of advertised bandwidth reaching HF / S3


@dataclass(frozen=True)
class Estimate:
    startup_minutes: float
    startup_cost: float
    hourly: float
    session_hours: float
    session_cost: float  # startup + session_hours of use
    monthly_cost: float  # 22 working days at session_hours


def startup_minutes(download_gb: float, inet_down_mbps: float, *, cached: bool = False) -> float:
    if cached or inet_down_mbps <= 0:
        dl = 0.0 if cached else max(30.0, download_gb * 0.4)  # unknown bandwidth: pessimistic (~40 MB/s)
    else:
        dl = download_gb * 8000 / (inet_down_mbps * DOWNLOAD_EFFICIENCY) / 60
    load = max(MIN_LOAD_MIN, download_gb / 100 * LOAD_MIN_PER_100GB)
    return PROVISION_MIN + dl + load


def estimate(offer: Offer, download_gb: float, session_hours: float, days_per_month: int = 22) -> Estimate:
    mins = startup_minutes(download_gb, offer.inet_down_mbps)
    startup_cost = mins / 60 * offer.dph_total
    per_session = startup_cost + session_hours * offer.dph_total
    return Estimate(mins, startup_cost, offer.dph_total, session_hours, per_session, per_session * days_per_month)


def session_cost(dph: float, start_epoch: float, now: float | None = None) -> float:
    return max(0.0, ((now or time.time()) - start_epoch) / 3600 * dph)


# --- ledger ---------------------------------------------------------------

def record_session(session_id: str, start_epoch: float, end_epoch: float, cost: float, model: str) -> None:
    paths.state_dir().mkdir(parents=True, exist_ok=True)
    with paths.ledger_file().open("a") as f:
        f.write(json.dumps({
            "session_id": session_id, "start": start_epoch, "end": end_epoch,
            "cost": round(cost, 4), "model": model,
        }) + "\n")


def month_to_date(now: float | None = None) -> tuple[float, int]:
    """(estimated spend, session count) for the current calendar month (UTC)."""
    f = paths.ledger_file()
    if not f.exists():
        return 0.0, 0
    ref = datetime.fromtimestamp(now or time.time(), timezone.utc)
    total, n = 0.0, 0
    for line in f.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        d = datetime.fromtimestamp(row["start"], timezone.utc)
        if (d.year, d.month) == (ref.year, ref.month):
            total += row["cost"]; n += 1
    return total, n
