from coder import costs
from coder.config import Config
from coder.models import load_registry
from coder.offers import build_query, filter_offers, rank_offers, required_disk_gb
from conftest import make_offer


def _q(model="qwen3-coder-next"):
    _, v = load_registry().get(model).variant(None)
    return build_query(Config(), v), v


def test_filters_vram_gpu_count_price_reliability():
    q, _ = _q()
    good = make_offer(1)
    offers = [
        good,
        make_offer(2, n=1, ram=141),            # single GPU not allowed for Qwen3-Coder-Next
        make_offer(3, n=2, ram=24),             # 48 GB total: too small
        make_offer(4, price=9.0),               # over max price
        make_offer(5, rel=0.80),                # unreliable
        make_offer(6, geo="Berlin, DE", disk=100),  # disk too small
        make_offer(7, cpu=8),                   # too little system RAM
    ]
    assert filter_offers(offers, q) == [good]


def test_country_and_gpu_name_filters():
    q, _ = _q()
    q.countries = ["DE"]
    assert filter_offers([make_offer(1), make_offer(2, geo="Berlin, DE")], q)[0].id == 2
    q.countries, q.gpu_names = [], ["a100"]
    assert [o.id for o in filter_offers([make_offer(1), make_offer(2, gpu="A100_SXM4")], q)] == [2]


def test_ranking_prefers_value_and_reliability():
    cheap_flaky = make_offer(1, price=2.0, rel=0.97)
    pricier_solid = make_offer(2, price=2.1, rel=0.999)
    cheapest = make_offer(3, price=1.5, rel=0.99)
    ranked = rank_offers([cheap_flaky, pricier_solid, cheapest], [])
    assert ranked[0].id == 3
    assert rank_offers([make_offer(1, gpu="RTX 6000Ada", price=2.0), make_offer(2, gpu="H100", price=1.9)], ["H100"])[0].id == 2


def test_disk_requirement_scales_with_checkpoint():
    _, fp8 = load_registry().get("qwen3-coder-next").variant("fp8")
    _, small = load_registry().get("devstral-small-2").variant(None)
    assert required_disk_gb(fp8) > required_disk_gb(small) > 45


def test_cost_estimate():
    o = make_offer(price=3.0, down=1000)
    e = costs.estimate(o, 85, 3.0)
    assert 10 < e.startup_minutes < 40
    assert abs(e.session_cost - (e.startup_cost + 9.0)) < 1e-9
    assert abs(e.monthly_cost - e.session_cost * 22) < 1e-9
    assert costs.estimate(make_offer(down=0), 85, 3).startup_minutes > costs.estimate(o, 85, 3).startup_minutes


def test_ledger_month_to_date(isolated_home):
    import time
    now = time.time()
    costs.record_session("a", now - 7200, now - 3600, 4.0, "m")
    costs.record_session("b", now - 3600, now, 6.0, "m")
    costs.record_session("old", now - 86400 * 100, now - 86400 * 100 + 10, 99.0, "m")
    total, n = costs.month_to_date(now)
    assert (total, n) == (10.0, 2)


def test_excluded_gpus_are_filtered():
    q, _ = _q("qwen3-coder-30b-a3b")
    q.exclude_gpus = ["CMP"]
    ids = [o.id for o in filter_offers([make_offer(1, n=1, ram=66, gpu="CMP 170HX"), make_offer(2, n=1, ram=80, gpu="A100")], q)]
    assert ids == [2]


def test_include_gpus_allow_list_from_config():
    cfg = Config()
    cfg.offers.include_gpus = ["A100", "4090"]
    _, v = load_registry().get("qwen3-coder-30b-a3b").variant(None)
    q = build_query(cfg, v)
    ids = [o.id for o in filter_offers([make_offer(1, n=1, ram=48, gpu="Quadro RTX 8000"), make_offer(2, n=1, ram=80, gpu="A100 SXM4"), make_offer(3, n=1, ram=24, gpu="RTX 4090")], q)]
    assert ids == [2]  # 4090 (24 GB) is below the checkpoint's per-GPU minimum; RTX 8000 is not on the list
