#!/usr/bin/env python
"""Check the api-key path end to end: a llama-server started with --api-key must answer when the client sends the bearer
token (this is how GINOmoto talks to a jev served by somebody else) and refuse without it."""
from __future__ import annotations

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from jev.model_llama import JevGGUF  # noqa: E402

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8098"
KEY = sys.argv[2] if len(sys.argv) > 2 else "s3cret"

pair = ("A man is playing a guitar.", ["Someone is making music."])

try:
    jev = JevGGUF(url=URL, api_key=KEY)
    p = jev.probs(*pair)
    print(f"com chave   -> OK  p_entailment={p[0, 1]:.2f}  (marker {jev.marker[:24]}...)")
except Exception as e:
    print("com chave   -> FALHOU:", e)

try:
    jev = JevGGUF(url=URL)
    p = jev.probs(*pair)
    print(f"sem chave   -> DEVERIA RECUSAR, mas respondeu p_ent={p[0, 1]:.2f}")
except Exception as e:
    print("sem chave   -> recusado como esperado:", type(e).__name__, e)
