"""The jev brain, served by llama.cpp instead of PyTorch.

Why: Qwen3.5 is a hybrid (24 of 32 layers are Gated DeltaNet, not softmax attention). On this laptop the PyTorch
fallback for the gated delta rule (transformers has no fused kernel without flash-linear-attention, which needs a GPU)
runs at ~2.4 s/token — measured, see logs/bench2.log — so the whole plan is unusable in `transformers`. llama.cpp has
proper CPU/Vulkan kernels for the architecture, so the backbone runs there (GGUF) and the NLI head stays here: the
server is started with `--pooling none`, which returns the per-token hidden states, and the 3-class decision is a
3 x hidden matmul plus a softmax on the last token — exactly what the HF checkpoint's `score` head did.

`probs` / `probs_img` / `calls` / `secs` / `report` are the same interface as jev.model.Jev, so the planner and the chat
layer do not know which backend they are talking to.
"""
from __future__ import annotations

import base64
import json
import time
import urllib.request

import numpy as np

DEFAULT_TEMPLATE = "Premise: {premise}\nHypothesis: {hypothesis}"
MARKER = "<__media__>"  # mtmd_default_marker(): the placeholder mtmd replaces with the image tokens


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


class JevGGUF:
    def __init__(self, url: str = "http://127.0.0.1:8099", head: str = "models/jev-head.npz",
                 template: str = DEFAULT_TEMPLATE, timeout: int = 900, bs_img: int = 2, api_key: str = ""):
        z = np.load(head)
        self.W = z["score_weight"].astype(np.float32)          # (3, hidden)
        self.labels = [str(x) for x in z["labels"]]
        self.url, self.template, self.timeout = url.rstrip("/"), template, timeout
        self.marker = self._fetch_marker()  # llama-server randomises it per process; /props publishes it
        self.calls, self.secs, self.load_s, self.bs_img = 0, 0.0, 0.0, bs_img
        self.api_key = api_key
        self.patch_embed_patched = False

    def _fetch_marker(self) -> str:
        try:
            rq = urllib.request.Request(self.url + "/props", headers=self._headers())
            props = json.loads(urllib.request.urlopen(rq, timeout=30).read())
            return props.get("media_marker") or MARKER
        except Exception:
            return MARKER

    # ------------------------------------------------------------------ transport
    def _headers(self):
        h = {"content-type": "application/json"}
        if self.api_key:                      # llama-server --api-key: for a jev served by somebody else
            h["authorization"] = f"Bearer {self.api_key}"
        return h

    def _post(self, content, timeout=None):
        body = json.dumps({"content": content}).encode()
        req = urllib.request.Request(self.url + "/embedding", data=body, headers=self._headers())
        t0 = time.perf_counter()
        raw = json.loads(urllib.request.urlopen(req, timeout=timeout or self.timeout).read())
        self.calls += 1
        self.secs += time.perf_counter() - t0
        if isinstance(raw, dict):                     # error object
            raise RuntimeError(f"llama-server: {raw}")
        return raw                                    # list, one entry per prompt

    @staticmethod
    def _last_tokens(results):
        """Per prompt, the hidden state of its last token (this is what the NLI head pools)."""
        out = []
        for r in results:
            emb = r["embedding"] if isinstance(r, dict) else r
            arr = np.asarray(emb, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            out.append(arr[-1])
        return np.stack(out) if out else np.zeros((0, self.W.shape[1]), dtype=np.float32)

    # ------------------------------------------------------------------ api
    def probs(self, premise: str, hyps: list[str], pair_texts: list[str] | None = None) -> np.ndarray:
        texts = pair_texts or [self.template.format(premise=premise, hypothesis=h) for h in hyps]
        H = self._last_tokens(self._post(texts))
        return _softmax(H @ self.W.T)

    def probs_img(self, image, premise_tpl: str, hyps: list[str]) -> np.ndarray:
        path = image if isinstance(image, str) else getattr(image, "filename", None)
        if not path:
            raise ValueError("probs_img needs a path to a PNG frame")
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        prompt = premise_tpl.format(img=self.marker)
        content = [{"prompt_string": self.template.format(premise=prompt, hypothesis=h), "multimodal_data": [b64]}
                   for h in hyps]
        rows = []
        for i in range(0, len(content), self.bs_img):   # a batch bigger than n_batch is refused by the server
            rows.append(self._last_tokens(self._post(content[i:i + self.bs_img])))
        H = np.concatenate(rows, 0) if rows else np.zeros((0, self.W.shape[1]), dtype=np.float32)
        return _softmax(H @ self.W.T)

    def rerank(self, premise: str, options: list[str], hyp_fmt: str = "The correct answer is: {}") -> int:
        return int(self.probs(premise, [hyp_fmt.format(o) for o in options])[:, 1].argmax())

    def report(self) -> str:
        return (f"{self.calls} model calls, {self.secs / max(1, self.calls) * 1000:.0f} ms/call, "
                f"{self.secs:.1f} s of inference (llama.cpp, pooling=none + numpy NLI head)")
