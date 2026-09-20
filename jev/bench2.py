#!/usr/bin/env python
"""Where does the time go? Loads the jev checkpoint once and times single forwards at three sequence lengths, printing
the model's layer types, so it is clear whether the cost is prefill (matmul) or the per-layer recurrent/conv path.

    ./venv/bin/python jev/bench2.py 2>&1 | tee logs/bench2.log
"""
from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jev.model import Jev  # noqa: E402


def main():
    t0 = time.time()
    jev = Jev("models/openjev/qwen3.5-4b-nli-v2", img_processor=None, threads=int(os.environ.get("THREADS", 10)), max_len=1024)
    print(f"load {time.time() - t0:.1f}s threads {torch.get_num_threads()}", flush=True)
    cfg = jev.model.config
    print("config:", {k: getattr(cfg, k) for k in ("hidden_size", "num_hidden_layers", "num_attention_heads", "vocab_size") if hasattr(cfg, k)}, flush=True)
    tc = cfg.get_text_config()
    lt = getattr(tc, "layer_types", None) or getattr(tc, "linear_attn_config", None)
    print("text config layer_types:", lt, flush=True)

    base = "Premise: Minecraft survival. Inventory counts: 0 logs, 0 planks, 0 sticks. Hypothesis: The player has at least 4 logs."
    toks = jev.tok(base, return_tensors="pt")["input_ids"]
    print("base sequence:", toks.shape[1], "tokens", flush=True)
    for n in (16, 64, 160, 320):
        ids = torch.cat([toks[:, :1], toks[:, 1:n - 1].repeat(1, 400)[:, :n - 2], toks[:, -1:]], 1)
        x = ids.expand(1, -1)
        jev._forward(["x"])  # unrelated warmup of the graph
        enc = {"input_ids": x, "attention_mask": torch.ones_like(x)}
        t0 = time.time()
        jev.model(**enc)
        dt = time.time() - t0
        print(f"  seq {n:4d} tokens  {dt:7.2f}s  {dt / n * 1000:7.1f} ms/token", flush=True)
    t0 = time.time()
    jev.model(**{"input_ids": x, "attention_mask": torch.ones_like(x)})
    print(f"  repeat seq {x.shape[1]} tokens: {time.time() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
