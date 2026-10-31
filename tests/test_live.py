"""Optional integration tests against the real Vast.ai API.

Never run by default (`-m 'not live'` in pytest config). Enable read-only checks with:
    CODER_LIVE_TESTS=1 VAST_API_KEY=... pytest -m live
The billable create/destroy test additionally needs CODER_LIVE_BILLABLE=1 and rents the
cheapest verified machine for a few seconds.
"""
import os
import time

import pytest

from coder.config import Config
from coder.models import load_registry
from coder.offers import build_query, filter_offers
from coder.providers.base import InstanceSpec
from coder.providers.vast import VastProvider

pytestmark = [pytest.mark.live, pytest.mark.skipif(not os.environ.get("CODER_LIVE_TESTS"), reason="set CODER_LIVE_TESTS=1")]


@pytest.fixture
def live():
    key = os.environ.get("REAL_VAST_API_KEY") or os.environ.get("VAST_API_KEY")
    if not key or key == "test-key":
        pytest.skip("no real VAST_API_KEY")
    return VastProvider(key)


def test_search_and_billing_readonly(live):
    _, v = load_registry().get("qwen3-coder-30b-a3b").variant(None)
    q = build_query(Config(), v)
    offers = filter_offers(live.search_offers(q), q)
    assert all(o.total_vram_gb >= v.min_total_vram_gb for o in offers)
    assert live.get_billing() is not None
    assert isinstance(live.list_managed_instances(), list)


@pytest.mark.skipif(not os.environ.get("CODER_LIVE_BILLABLE"), reason="billable: set CODER_LIVE_BILLABLE=1")
def test_create_and_destroy_cheapest(live):
    from coder.providers.base import OfferQuery
    offers = live.search_offers(OfferQuery([1], 8, 8, 0.5, 0.97, min_disk_gb=20))
    assert offers
    c = live.create_instance(offers[0], InstanceSpec("ubuntu:22.04", "coder-managed:live-test", 16))
    try:
        assert live.get_instance(c.id) is not None
    finally:
        live.destroy_instance(c.id)
    time.sleep(3)
    assert live.get_instance(c.id) is None
