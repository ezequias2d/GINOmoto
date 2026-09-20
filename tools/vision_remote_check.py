#!/usr/bin/env python
"""Checagem de visão contra um servidor jev remoto: manda um frame com 2 hipóteses e mostra o resultado/tempo."""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from jev.model_llama import JevGGUF  # noqa: E402
from jev.vision import PREMISE, SCENE  # noqa: E402

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8099"
KEY = sys.argv[2] if len(sys.argv) > 2 else "a-key"
FRAME = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shots/check.png")

jev = JevGGUF(url=URL, api_key=KEY)
print(f"servidor {URL} | marker {jev.marker}")
hyps = [h for h, _, _ in SCENE[:2]]
t0 = time.perf_counter()
try:
    p = jev.probs_img(FRAME, PREMISE, hyps)
    dt = time.perf_counter() - t0
    print(f"2 hipóteses com imagem: {dt:.1f}s")
    for (h, tag, steer), row in zip(SCENE[:2], p):
        print(f"   p_ent={row[1]:.2f} p_con={row[0]:.2f} [{tag}] {h}")
except Exception as e:
    print(f"FALHOU em {time.perf_counter() - t0:.1f}s: {type(e).__name__}: {e}")
