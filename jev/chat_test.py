#!/usr/bin/env python
"""How well does the jev model understand the player's chat orders? Prints the picked intent, the entailment score and
the keyword fallback, for Portuguese (what players actually type) and English commands.

    ./venv/bin/python jev/chat_test.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jev.backend import make_jev  # noqa: E402
from jev.chat import Chat  # noqa: E402

CASES = [
    ("pega madeira pra mim", "mine_logs"),
    ("corta umas arvore ai", "mine_logs"),
    ("minera pedra", "mine_stone"),
    ("pega ferro", "mine_iron"),
    ("me segue", "follow"),
    ("vem aqui", "come"),
    ("para", "stop"),
    ("fica quieto um pouco", "stop"),
    ("mata esses zumbi", "attack"),
    ("faz uma torre pra eu subir", "tower"),
    ("cava pra baixo", "dig_down"),
    ("me dá 3 madeira", "give"),
    ("olha pra mim", "look"),
    ("o que voce tem no inventario?", "status"),
    ("eae jev", "greet"),
    ("vai pra 120 -400", "goto"),
    ("boa, mandou bem", "praise"),
    ("cala a boca", "quiet"),
    ("explora aí", "explore"),
    ("faz um picareta de ferro", ("goal", "craft")),
    ("chop some wood", "mine_logs"),
    ("follow me", "follow"),
    ("stop it", "stop"),
    ("mine iron ore", "mine_iron"),
    ("make me an iron pickaxe", ("goal", "craft")),
    ("dig straight down", "dig_down"),
    ("build a tower", "tower"),
    ("drop some wood for me", "give"),
    ("how are you doing?", "greet"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="llama", choices=["llama", "torch"])
    ap.add_argument("--server-url", default="http://127.0.0.1:8099")
    ap.add_argument("--head", default="models/jev-head.npz")
    ap.add_argument("--ckpt", default="models/openjev/qwen3.5-4b-nli-v2")
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--out", default="logs/chat_test.json")
    ap.add_argument("--full-intents", action="store_true", help="score all 19 intents per message")
    ap.add_argument("--min-conf", type=float, default=0.55)
    ap.add_argument("--min-margin", type=float, default=0.05)
    args = ap.parse_args()

    jev = make_jev(args.backend, url=args.server_url, head=args.head, ckpt=args.ckpt, threads=args.threads)
    chat = Chat(jev, min_conf=args.min_conf, min_margin=args.min_margin, shorten=not args.full_intents)
    ok, rows, t0 = 0, [], time.time()
    for msg, want in CASES:
        want_ok = want if isinstance(want, tuple) else (want,)
        r = chat.interpret(msg)
        hit = r["intent"] in want_ok and (r["target"] == want[1] if isinstance(want, tuple) else True)
        ok += hit
        rows.append({**r, "want": want_ok, "hit": bool(hit)})
        print(f"{'OK ' if hit else 'MISS'} {msg:38s} -> {r['intent']:9s} ({r['used']:7s}) conf={r['conf']:.2f} "
              f"marg={r['margin']:.2f} target={r['target'] or '-'}  want={want_ok}  "
              f"[{', '.join(f'{n}:{p:.2f}' for n, p in r['ranked'][:3])}]", flush=True)
    print(f"\n{ok}/{len(CASES)} correct in {time.time() - t0:.0f}s; model calls {jev.calls}, "
          f"{1000 * jev.secs / max(1, jev.calls):.0f} ms/call", flush=True)
    print(f"pt-BR only: {sum(1 for r in rows[:20] if r['hit'])}/20", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"args": vars(args), "rows": rows, "score": f"{ok}/{len(CASES)}",
               "ms_per_call": 1000 * jev.secs / max(1, jev.calls)}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
