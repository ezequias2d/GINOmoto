#!/usr/bin/env python
"""Smoke test for the local jev checkpoint: does it load on this CPU, and does it answer text and image premises?

    ./venv/bin/python jev/smoke.py --frame shots/check.png
"""
from __future__ import annotations

import argparse
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jev.backend import make_jev  # noqa: E402
from jev.model import ENT  # noqa: E402
from jev.vision import PREMISE, SCENE  # noqa: E402


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="llama", choices=["llama", "torch"])
    ap.add_argument("--server-url", default="http://127.0.0.1:8099")
    ap.add_argument("--head", default="models/jev-head.npz")
    ap.add_argument("--ckpt", default="models/openjev/qwen3.5-4b-nli-v2")
    ap.add_argument("--img-processor", default="models/qwen35-4b-proc")
    ap.add_argument("--frame", default="shots/check.png")
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--image-size", type=int, nargs=2, default=[320, 240])
    args = ap.parse_args()

    print("loading (bf16, cpu)...", flush=True)
    t0 = time.time()
    jev = make_jev(args.backend, url=args.server_url, head=args.head, ckpt=args.ckpt, img_processor=args.img_processor,
                   threads=args.threads, image_size=args.image_size)
    print(f"loaded in {time.time() - t0:.1f}s  peak rss {rss_gb():.1f} GB  vision patched={jev.patch_embed_patched}", flush=True)
    print("labels:", getattr(jev, "labels", None) or jev.model.config.id2label, flush=True)

    pairs = [("A man is playing a guitar.", "Someone is making music.", "entailment"),
             ("A man is playing a guitar.", "The man is asleep in bed.", "contradiction"),
             ("A man is playing a guitar.", "The man is a professional musician.", "neutral")]
    t0 = time.time()
    p = jev.probs("A man is playing a guitar.", [h for _, h, _ in pairs])
    for (pr, h, gold), row in zip(pairs, p):
        got = ["contradiction", "entailment", "neutral"][int(row.argmax())]
        print(f"  txt  {h:40s} -> {got:14s} ({row[0]:.2f}/{row[1]:.2f}/{row[2]:.2f}) gold={gold} {'OK' if got == gold else 'WRONG':5s}", flush=True)
    print(f"text batch of 3: {time.time() - t0:.1f}s", flush=True)

    opts = ["chop wood", "mine stone", "swim in the ocean", "craft a pickaxe"]
    q = "The player is asking the bot to {}."
    t0 = time.time()
    rr = jev.rerank("Bot commands. The player says: 'pega madeira pra mim'.", [q.format(o) for o in opts])
    print(f"  rerank: {opts[rr]}  ({time.time() - t0:.1f}s)", flush=True)

    if os.path.exists(args.frame):
        print(f"vision on {args.frame} (resized to {tuple(args.image_size)}):", flush=True)
        t0 = time.time()
        cols = jev.probs_img(args.frame, PREMISE, [h for h, _, _ in SCENE])
        for (h, tag, steer), row in zip(SCENE, cols):
            print(f"  img  p_ent={row[1]:.2f} p_con={row[0]:.2f} [{tag:5s} steer {steer:3d}] {h}", flush=True)
        print(f"image batch of {len(SCENE)}: {time.time() - t0:.1f}s  -> verdict: "
              f"{SCENE[int(cols[:, ENT].argmax())][0]}", flush=True)

    print("model:", jev.report(), flush=True)
    print(f"peak rss {rss_gb():.1f} GB", flush=True)


if __name__ == "__main__":
    main()
