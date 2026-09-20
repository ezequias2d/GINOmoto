"""One place to pick the jev backend: `llama` (llama.cpp server, the one that works on this CPU) or `torch` (the HF
checkpoint in plain PyTorch — correct but ~2.4 s/token on this laptop because the gated delta rule has no fused kernel)."""
from __future__ import annotations


def make_jev(kind: str = "llama", **kw):
    if kind == "llama":
        from jev.model_llama import JevGGUF
        return JevGGUF(url=kw.get("url") or "http://127.0.0.1:8099", head=kw.get("head") or "models/jev-head.npz",
                       timeout=int(kw.get("timeout") or 900), api_key=kw.get("api_key") or "")
    from jev.model import Jev
    return Jev(kw.get("ckpt") or "models/openjev/qwen3.5-4b-nli-v2",
               img_processor=kw.get("img_processor") or "models/qwen35-4b-proc",
               threads=int(kw.get("threads") or 10), bs=int(kw.get("bs") or 8),
               image_size=tuple(kw.get("image_size") or (320, 240)))
