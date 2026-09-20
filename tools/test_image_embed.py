#!/usr/bin/env python
"""Send one image premise to llama-server and print what comes back (body, status, timing)."""
from __future__ import annotations

import base64
import json
import sys
import time
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8099"


def post(path, body, timeout=600):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        return r, time.perf_counter() - t0, 200
    except urllib.error.HTTPError as e:
        return e.read().decode()[:800], time.perf_counter() - t0, e.code


def main():
    frame = sys.argv[1] if len(sys.argv) > 1 else "shots/check.png"
    props = json.loads(urllib.request.urlopen(URL + "/props", timeout=30).read())
    marker = props.get("media_marker")
    print("media_marker:", marker)
    b64 = base64.b64encode(open(frame, "rb").read()).decode()
    print("frame:", frame, f"{len(b64) / 1024:.0f} KiB base64")

    text = ("Premise: A first-person Minecraft screenshot taken by a bot, the crosshair in the centre of the image: "
            f"{marker} Add nothing to the picture.\nHypothesis: A tree trunk with leaves is standing right in front of the bot.")
    body = {"content": [{"prompt_string": text, "multimodal_data": [b64]}]}
    r, dt, code = post("/embedding", body)
    print(f"image request -> HTTP {code} in {dt:.1f}s")
    if isinstance(r, str):
        print("body:", r)
        return
    emb = r[0]["embedding"]
    print("tokens:", len(emb), "dim:", len(emb[0]), "| last3:", [round(x, 3) for x in emb[-1][:3]])

    txt = "Premise: A man is playing a guitar.\nHypothesis: Someone is making music."
    r, dt, code = post("/embedding", {"content": [txt]})
    print(f"text request  -> HTTP {code} in {dt:.1f}s")


if __name__ == "__main__":
    main()
