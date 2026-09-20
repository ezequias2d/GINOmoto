#!/usr/bin/env python
"""Check the openjev checkpoint against the GGUF we produced: block count (32 + MTP?), vision tensors, block 32."""
from __future__ import annotations

import collections

import gguf
from safetensors import safe_open

SRC = "models/openjev/qwen3.5-4b-nli-v2/model.safetensors"
GGUF = "models/jev/jev-qwen35-4b-f16.gguf"

with safe_open(SRC, framework="pt") as f:
    ks = list(f.keys())
    blocks = collections.Counter(k.split(".layers.")[1].split(".")[0] for k in ks if k.startswith("model.language_model") and ".layers." in k)
    print("source blocks:", sorted(int(b) for b in blocks), "| n =", len(blocks))
    print("mtp/nextn tensors:", [k for k in ks if "mtp" in k.lower() or "nextn" in k.lower()][:5])
    print("visual tensors:", sum(1 for k in ks if k.startswith("model.visual")), "| total", len(ks))
    print("embed name:", [k for k in ks if "embed_tokens" in k][:2])

r = gguf.GGUFReader(GGUF)
blk = collections.Counter(t.name.split(".")[1] for t in r.tensors if t.name.startswith("blk."))
print("gguf blocks:", sorted(int(b) for b in blk), "| tensors in blk.32:", blk.get("32", 0))
print("gguf tensor count:", len(r.tensors))
