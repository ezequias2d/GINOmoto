#!/usr/bin/env python
"""Throughput micro-benchmark for the jev checkpoint on this CPU: text prefill, batch of hypotheses, image premise."""
from __future__ import annotations

import argparse
import os
import resource
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jev.model import Jev  # noqa: E402
from jev.vision import PREMISE, SCENE  # noqa: E402

rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/openjev/qwen3.5-4b-nli-v2")
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--frame", default="shots/check.png")
    args = ap.parse_args()

    t0 = time.time()
    jev = Jev(args.ckpt, img_processor=None, threads=args.threads)
    print(f"load {time.time() - t0:.1f}s  rss {rss():.2f} GB  threads {torch.get_num_threads()}", flush=True)

    prem = "Minecraft survival. Inventory counts: 0 logs, 0 planks. A log block is visible nearby."
    hyps = [f"The player has at least {n} logs." for n in (1, 2, 4, 8, 16, 32)]
    texts = [jev.template.format(premise=prem, hypothesis=h) for h in hyps]
    tok = jev.tok(texts[0], return_tensors="pt")
    print(f"one sequence: {tok['input_ids'].shape[1]} tokens", flush=True)

    for label, batch in (("1 hyp", texts[:1]), ("3 hyps", texts[:3]), ("6 hyps", texts[:6])):
        jev._forward(batch)  # warmup
        t0 = time.time()
        jev._forward(batch)
        dt = time.time() - t0
        nt = sum(len(jev.tok(t, add_special_tokens=False)["input_ids"]) for t in batch)
        print(f"  {label:7s} {nt:5d} tokens  {dt:6.2f}s  {nt / dt:6.1f} tok/s", flush=True)

    if os.path.exists(args.frame):
        for label, hs in (("img 3", SCENE[:3]), ("img 6", SCENE)):
            jev.probs_img(args.frame, PREMISE, [h for h, _, _ in hs[:1]][:1] and [h for h, _, _ in hs])  # warmup
            t0 = time.time()
            jev.probs_img(args.frame, PREMISE, [h for h, _, _ in hs])
            print(f"  {label:7s} {time.time() - t0:6.2f}s", flush=True)
    print(f"peak rss {rss():.2f} GB", flush=True)


if __name__ == "__main__":
    main()
