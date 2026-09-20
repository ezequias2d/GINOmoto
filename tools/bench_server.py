#!/usr/bin/env python
"""Where does the server time go: one prompt, a small batch, the full chat batch (19 intents) in one request."""
from __future__ import annotations

import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8099/embedding"
TPL = "Premise: {p}\nHypothesis: {h}"
HEADERS = {"content-type": "application/json", "user-agent": "GINOmoto/1.0"}


def post(content, timeout=1800):
    req = urllib.request.Request(URL, data=json.dumps({"content": content}).encode(), headers=HEADERS)
    t0 = time.perf_counter()
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return r, time.perf_counter() - t0


def main():
    global URL
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    import argparse
    from jev.chat import INTENTS, PREMISE  # noqa: E402
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", default="http://127.0.0.1:8099")
    ap.add_argument("--api-key", default="")
    args = ap.parse_args()
    URL = args.server_url.rstrip("/") + "/embedding"
    if args.api_key:
        HEADERS["authorization"] = f"Bearer {args.api_key}"
    print("servidor:", URL, flush=True)

    msg = "pega madeira pra mim"
    premise = PREMISE.format(msg=msg)
    texts = [TPL.format(p=premise, h=h) for _, h, _ in INTENTS]
    for label, n in (("1 prompt", 1), ("6 prompts", 6), ("19 prompts (full chat batch)", len(texts))):
        r, dt = post(texts[:n])
        ntok = sum(len(x["embedding"]) for x in r)
        print(f"  {label:30s} {dt:7.2f}s  ({dt / n:5.2f}s per sequence, {ntok} tokens returned)", flush=True)


if __name__ == "__main__":
    main()
