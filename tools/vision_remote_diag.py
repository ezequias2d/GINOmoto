#!/usr/bin/env python
"""Diagnóstico do caminho de visão em um servidor jev remoto: o que /props devolve e por que a requisição com imagem
falha (corpo do erro, não só o código)."""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8099").rstrip("/")
KEY = sys.argv[2] if len(sys.argv) > 2 else "a-key"
FRAME = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shots/check.png")
H = {"content-type": "application/json", "user-agent": "GINOmoto/1.0", "authorization": f"Bearer {KEY}"}


def show(label, req):
    try:
        r = urllib.request.urlopen(req, timeout=300)
        body = r.read()
        print(f"{label}: HTTP {r.status}, {len(body)} bytes")
        return body
    except urllib.error.HTTPError as e:
        print(f"{label}: HTTP {e.code} -> {e.read().decode()[:400]}")
    except Exception as e:
        print(f"{label}: {type(e).__name__}: {e}")
    return None


b = show("GET /props", urllib.request.Request(URL + "/props", headers=H))
if b:
    d = json.loads(b)
    print("   media_marker:", d.get("media_marker"))
    print("   n_ctx (slot):", d.get("default_generation_settings", {}).get("n_ctx"), "| modalities:", d.get("modalities"))

b64 = base64.b64encode(open(FRAME, "rb").read()).decode()
marker = (json.loads(b).get("media_marker") if b else None) or "<__media__>"
text = f"Premise: A first-person Minecraft screenshot: {marker} Hypothesis: There is a tree in front of the bot."
body = json.dumps({"content": [{"prompt_string": text, "multimodal_data": [b64]}]}).encode()
show("POST /embedding (imagem)", urllib.request.Request(URL + "/embedding", data=body, headers=H))

body = json.dumps({"content": ["Premise: teste\nHypothesis: teste"]}).encode()
show("POST /embedding (texto)", urllib.request.Request(URL + "/embedding", data=body, headers=H))
