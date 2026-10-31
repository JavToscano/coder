import json

import httpx
import pytest

from coder.errors import OfferUnavailable, ProviderError
from coder.providers.base import InstanceSpec, InstanceState, OfferQuery
from coder.providers.vast import VastProvider
from conftest import make_offer

Q = OfferQuery(gpu_counts=[2, 4], min_gpu_vram_gb=40, min_total_vram_gb=150, max_hourly_price=4, min_reliability=0.97, min_cpu_ram_gb=32, min_disk_gb=200)


def provider(handler, **kw):
    sleeps = []
    p = VastProvider("k", transport=httpx.MockTransport(handler), sleep=sleeps.append, **kw)
    p.sleeps = sleeps
    return p


def test_search_request_and_parse():
    seen = {}

    def h(req):
        seen["auth"], seen["body"], seen["path"] = req.headers["authorization"], json.loads(req.content), req.url.path
        return httpx.Response(200, json={"offers": [{
            "id": 77, "gpu_name": "H100 SXM", "num_gpus": 2, "gpu_ram": 81559, "cpu_ram": 257000, "disk_space": 800,
            "dph_total": 3.2, "reliability": 0.995, "geolocation": "Texas, US", "inet_down": 3000}]})

    offers = provider(h).search_offers(Q)
    assert seen["auth"] == "Bearer k" and seen["path"] == "/api/v0/bundles/"
    b = seen["body"]
    assert b["num_gpus"] == {"in": [2, 4]} and b["gpu_ram"] == {"gte": 40000} and b["dph_total"] == {"lte": 4}
    assert b["type"] == "on-demand" and b["rentable"] == {"eq": True}
    o = offers[0]
    assert (o.id, o.num_gpus, round(o.gpu_ram_gb), o.country, o.total_vram_gb > 160) == (77, 2, 82, "US", True)


def test_search_retries_type_spelling_on_400():
    types = []

    def h(req):
        types.append(json.loads(req.content)["type"])
        return httpx.Response(400, text="bad type") if len(types) == 1 else httpx.Response(200, json={"offers": []})

    assert provider(h).search_offers(Q) == [] and types == ["on-demand", "ondemand"]


def test_create_instance_body_and_response():
    seen = {}

    def h(req):
        seen.update(path=req.url.path, method=req.method, body=json.loads(req.content))
        return httpx.Response(200, json={"success": True, "new_contract": 555, "instance_api_key": "ik"})

    c = provider(h).create_instance(make_offer(id=9), InstanceSpec("vllm/vllm-openai:latest", "coder-managed:abc", 200))
    assert (seen["method"], seen["path"]) == ("PUT", "/api/v0/asks/9/")
    assert seen["body"] == {"image": "vllm/vllm-openai:latest", "label": "coder-managed:abc", "disk": 200, "runtype": "ssh_direct"}
    assert (c.id, c.instance_api_key) == (555, "ik")


def test_create_offer_gone():
    p = provider(lambda r: httpx.Response(410, json={"success": False, "error": "gone", "msg": "x"}))
    with pytest.raises(OfferUnavailable):
        p.create_instance(make_offer(), InstanceSpec("i", "l", 10))


def test_instance_state_and_ssh_mapping():
    body = {"instances": {"id": 5, "actual_status": "running", "cur_state": "running", "public_ipaddr": "9.9.9.9\n",
                          "ports": {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "41022"}]}, "dph_total": 2.5,
                          "gpu_name": "A100", "num_gpus": 2, "gpu_ram": 80000, "label": "coder-managed:x", "start_date": 1.7e9}}
    i = provider(lambda r: httpx.Response(200, json=body)).get_instance(5)
    assert i.state is InstanceState.RUNNING and (i.ssh.host, i.ssh.port) == ("9.9.9.9", 41022)

    def state(**kw):
        return provider(lambda r: httpx.Response(200, json={"instances": {"id": 1, **kw}})).get_instance(1).state
    assert state(actual_status="loading", cur_state="running") is InstanceState.PENDING
    assert state(actual_status=None, cur_state="running") is InstanceState.PENDING
    assert state(actual_status="exited", cur_state="stopped") is InstanceState.STOPPED
    assert state(actual_status="offline", cur_state="running") is InstanceState.FAILED


def test_get_instance_404_is_none():
    assert provider(lambda r: httpx.Response(404, json={"msg": "nope"})).get_instance(1) is None


def test_list_managed_only_returns_tagged_and_paginates():
    pages = {None: {"instances": [{"id": 1, "label": "coder-managed:a"}, {"id": 2, "label": "my-training-run"}], "next_token": "t"},
             "t": {"instances": [{"id": 3, "label": None}, {"id": 4, "label": "coder-managed:b"}]}}
    p = provider(lambda r: httpx.Response(200, json=pages[r.url.params.get("after_token")]))
    assert [i.id for i in p.list_managed_instances()] == [1, 4]


def test_destroy_stop_start_attach_requests():
    calls = []
    p = provider(lambda r: (calls.append((r.method, r.url.path, r.content)), httpx.Response(200, json={"success": True}))[1])
    p.destroy_instance(7); p.stop_instance(7); p.start_instance(7); p.attach_ssh_key(7, "ssh-ed25519 AAA\n")
    assert calls[0][:2] == ("DELETE", "/api/v0/instances/7")
    assert json.loads(calls[1][2]) == {"state": "stopped"} and json.loads(calls[2][2]) == {"state": "running"}
    assert calls[3][:2] == ("POST", "/api/v0/instances/7/ssh") and json.loads(calls[3][2]) == {"ssh_key": "ssh-ed25519 AAA"}


def test_rate_limit_backoff_then_success_and_exhaustion():
    n = {"c": 0}

    def h(req):
        n["c"] += 1
        return httpx.Response(429, text="too frequent") if n["c"] < 3 else httpx.Response(200, json={"success": True})

    p = provider(h)
    p.destroy_instance(1)
    assert n["c"] == 3 and len(p.sleeps) == 2 and p.sleeps[1] > p.sleeps[0]
    with pytest.raises(ProviderError, match="retries"):
        provider(lambda r: httpx.Response(429), max_retries=2).destroy_instance(1)


def test_auth_error_message():
    with pytest.raises(ProviderError, match="API key"):
        provider(lambda r: httpx.Response(401)).get_billing()


def test_billing():
    assert provider(lambda r: httpx.Response(200, json={"balance": 12.5})).get_billing().balance_usd == 12.5
