"""The jev brain: Qwen3.5-4B fine-tuned as an NLI cross-encoder (AlexWortega/openjev), running on this machine's CPU.

Nothing here generates text. The only primitive is `probs(premise, hypotheses) -> P([contradiction, entailment, neutral])`
and, with `probs_img`, the same thing with a screenshot in the premise (image tokens go through the Qwen3.5 vision
tower, which the NLI fine-tune left untouched). Every decision the bot makes is an entailment check.

The class follows the repo's `modeling_openjev.py` / `code/doom_vision.py` (MIT), with three additions for CPU use:
bf16 with a patched fp32 patch-embed, batching with a small cache, and per-call timing.
"""
from __future__ import annotations

import time

import numpy as np
import torch

CON, ENT, NEU = 0, 1, 2
DEFAULT_TEMPLATE = "Premise: {premise}\nHypothesis: {hypothesis}"


class FastPatchEmbed(torch.nn.Module):
    """The Qwen3.5 vision patch embed is a Conv3d with kernel == stride; bf16 conv3d is slow (and poorly supported)
    on CPU, the fp32 conv is fast and numerically closer to the reference. `weight` stays bf16 because the caller
    reads `proj.weight.dtype` to cast its input."""

    def __init__(self, conv: torch.nn.Module):
        super().__init__()
        self.weight, self.bias, self.stride = conv.weight, conv.bias, conv.stride

    def forward(self, x):
        y = torch.nn.functional.conv3d(x.float(), self.weight.float(), self.bias.float(), stride=self.stride)
        return y.to(self.weight.dtype)


class Jev:
    """Text (and image) entailment scorer, CPU, no gradients."""

    def __init__(self, ckpt: str, img_processor: str | None = None, threads: int | None = None,
                 bs: int = 8, max_len: int = 512, image_size=(320, 240)):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if threads:
            torch.set_num_threads(threads)
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        t0 = time.perf_counter()
        self.model = AutoModelForSequenceClassification.from_pretrained(ckpt, dtype=torch.bfloat16)
        self.model.eval()
        self.load_s = time.perf_counter() - t0
        tc = self.model.config.get_text_config()
        if tc.pad_token_id is None:
            tc.pad_token_id = self.tok.pad_token_id
        self.template = getattr(self.model.config, "nli_template", None) or DEFAULT_TEMPLATE
        self.img_id = self.tok.convert_tokens_to_ids("<|image_pad|>")
        self.bs, self.max_len, self.image_size = bs, max_len, image_size
        self.ip = None
        self.patch_embed_patched = False
        if img_processor:
            from transformers import AutoImageProcessor
            self.ip = AutoImageProcessor.from_pretrained(img_processor)
            try:  # the vision tower only exists on the v2 checkpoint
                self.model.model.visual.patch_embed.proj = FastPatchEmbed(self.model.model.visual.patch_embed.proj)
                self.patch_embed_patched = True
            except AttributeError as e:
                print(f"[jev] no vision tower patched ({e}) — image scoring unavailable", flush=True)
                self.ip = None
        self.calls, self.secs, self.cache = 0, 0.0, {}

    # ---------------------------------------------------------------- text
    @torch.inference_mode()
    def _forward(self, texts, extra=None):
        enc = self.tok(texts, truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
        if extra is None:
            return self.model(**enc).logits.float()
        kw = dict(extra)
        kw.update(enc)
        try:
            out = self.model(**kw).logits.float()
        except TypeError:  # older signatures without mm_token_type_ids
            kw.pop("mm_token_type_ids", None)
            out = self.model(**kw).logits.float()
        return out

    def probs(self, premise: str, hyps: list[str], pair_texts: list[str] | None = None) -> np.ndarray:
        """[len(hyps), 3] softmax over (contradiction, entailment, neutral)."""
        texts = pair_texts or [self.template.format(premise=premise, hypothesis=h) for h in hyps]
        out, t0 = [], time.perf_counter()
        for i in range(0, len(texts), self.bs):
            out.append(torch.softmax(self._forward(texts[i:i + self.bs]), -1).numpy())
        self.calls += 1
        self.secs += time.perf_counter() - t0
        return np.concatenate(out, 0)

    # ---------------------------------------------------------------- images
    def probs_img(self, image, premise_tpl: str, hyps: list[str]) -> np.ndarray:
        """The premise carries the frame: `{img}` is replaced by the special image tokens. One copy of the image per
        hypothesis (the cross-encoder has to see them as independent sequences)."""
        if self.ip is None:
            raise RuntimeError("no image processor loaded — pass img_processor to Jev()")
        from PIL import Image
        img = image if isinstance(image, Image.Image) else Image.open(image)
        img = img.convert("RGB").resize(self.image_size)
        vis = self.ip(images=[img], return_tensors="pt")
        n = int(vis["image_grid_thw"].prod()) // (self.ip.merge_size ** 2)
        tokens = "<|vision_start|>" + "<|image_pad|>" * n + "<|vision_end|>"
        texts = [self.template.format(premise=premise_tpl.format(img=tokens), hypothesis=h) for h in hyps]
        enc = self.tok(texts, truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
        extra = {
            "mm_token_type_ids": (enc["input_ids"] == self.img_id).long(),
            "pixel_values": vis["pixel_values"].repeat(len(hyps), *([1] * (vis["pixel_values"].dim() - 1))),
            "image_grid_thw": vis["image_grid_thw"].repeat(len(hyps), 1),
        }
        t0 = time.perf_counter()
        logits = self._forward(texts, extra=extra)
        self.calls += 1
        self.secs += time.perf_counter() - t0
        return torch.softmax(logits, -1).numpy()

    # ---------------------------------------------------------------- helpers
    def rerank(self, premise: str, options: list[str], hyp_fmt: str = "The correct answer is: {}") -> int:
        """Zero-shot multiple choice: the option with the highest P(entailment)."""
        return int(self.probs(premise, [hyp_fmt.format(o) for o in options])[:, ENT].argmax())

    def report(self) -> str:
        return (f"{self.calls} model calls, {self.secs / max(1, self.calls) * 1000:.0f} ms/call, "
                f"{self.secs:.1f} s of inference, weights loaded in {self.load_s:.1f} s")
