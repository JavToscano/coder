#!/usr/bin/env python3
"""Runs ON the GPU instance. Destroys *this* instance when a deadline passes or (opt-in)
when no inference requests were served for `idle_minutes`. Stdlib only.

It is the safety net for a crashed CLI, a sleeping Mac or a dropped network: it does not
depend on anything running locally.

Request tracking: counts access-log lines for POST /v1/... in the server log (a request is
logged when it completes) and reads the engine's Prometheus gauges for in-flight requests.
If the in-flight gauge cannot be read, an in-flight streaming request longer than the idle
timeout would be invisible, hence idle shutdown is opt-in.
"""
import json
import re
import sys
import time
import urllib.request

CFG = json.load(open("/root/coder/watchdog.json"))
KEY = open("/root/coder/provider.key").read().strip()
LOG = "/root/coder/server.log"
POLL = 30
RUN = re.compile(r"^(?:vllm:num_requests_running|sglang:num_running_reqs)(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", re.M)
WAIT = re.compile(r"^(?:vllm:num_requests_waiting|sglang:num_queue_reqs)(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", re.M)


def say(msg):
    print(time.strftime("%F %T"), msg, flush=True)


def destroy():
    req = urllib.request.Request(
        "https://console.vast.ai/api/v0/instances/%d" % CFG["instance_id"],
        method="DELETE", headers={"Authorization": "Bearer " + KEY},
    )
    for _ in range(10):
        try:
            urllib.request.urlopen(req, timeout=30).read()
            say("destroy requested")
            return
        except Exception as e:  # keep trying: this is the last line of defence
            say("destroy failed: %r" % (e,))
            time.sleep(15)


def request_count():
    try:
        with open(LOG, errors="replace") as f:
            return sum(1 for line in f if "POST /v1/" in line)
    except OSError:
        return 0


def in_flight():
    try:
        text = urllib.request.urlopen("http://127.0.0.1:%d/metrics" % CFG["port"], timeout=5).read().decode()
    except Exception:
        return None
    run = RUN.findall(text)
    if not run:
        return None
    return sum(float(x) for x in run) + sum(float(x) for x in WAIT.findall(text))


def main():
    last_count, last_activity = request_count(), time.time()
    while True:
        now = time.time()
        if now >= CFG["deadline_epoch"]:
            say("deadline reached")
            return destroy()
        if CFG.get("idle_minutes"):
            count, busy = request_count(), in_flight()
            if count != last_count or (busy or 0) > 0:
                last_count, last_activity = count, now
            elif now - last_activity >= CFG["idle_minutes"] * 60 and now > CFG.get("idle_not_before_epoch", 0):
                say("idle for %d min" % CFG["idle_minutes"])
                return destroy()
        time.sleep(POLL)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        say("watchdog crashed: %r" % (e,))
        sys.exit(1)
