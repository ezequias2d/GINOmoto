#!/usr/bin/env python
"""Convert the openjev checkpoint (Qwen3.5-4B seq-classification) to GGUF for llama.cpp.

The HF checkpoint is a *sequence classification* fine-tune: same Qwen3.5-4B backbone, `score.weight` (3 x hidden)
instead of `lm_head` (tie_word_embeddings=True), plus the untouched Qwen3.5 vision tower. llama.cpp's converter
already knows the Qwen3.5 text backbone and the Qwen3-VL vision tower, it just does not know this architecture name,
so:

  * the config is flattened (the text config lives under `text_config` in the seq-cls checkpoint) into a staging dir
    that symlinks the real weights;
  * `score.weight` — which the converter would refuse to map — is dropped from the GGUF and saved as
    models/jev-head.npz, because the NLI decision itself (softmax over the 3 classes) is applied in numpy on the
    pooled hidden state that llama.cpp returns for `--pooling none`;
  * a subclass is registered for `Qwen3_5ForSequenceClassification` so the existing Qwen3.5 text/vl code runs unchanged.

    python tools/convert_jev.py --text --out models/jev/jev-qwen35-4b-f16.gguf
    python tools/convert_jev.py --mmproj --out models/jev/mmproj-jev-qwen35-4b-f16.gguf
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import shutil
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA = os.environ.get("LLAMA_CPP", os.path.expanduser("~/llama.cpp"))
SRC = os.path.join(HERE, "models/openjev/qwen3.5-4b-nli-v2")
PROC = os.path.join(HERE, "models/qwen35-4b-proc")
STAGE = os.path.join(HERE, "models/jev-gguf-src")
HEAD = os.path.join(HERE, "models/jev-head.npz")
ARCH = "Qwen3_5ForSequenceClassification"


def stage(force: bool = False):
    """A directory the stock converter understands: flat Qwen3.5 text config + symlinked weights."""
    os.makedirs(STAGE, exist_ok=True)
    cfg = json.load(open(os.path.join(SRC, "config.json")))
    flat = dict(cfg.get("text_config") or {})
    flat.update({k: v for k, v in cfg.items() if k not in ("text_config", "vision_config")})
    flat["model_type"] = "qwen3_5"
    flat["architectures"] = [ARCH]
    flat["tie_word_embeddings"] = True
    for extra in (os.path.join(SRC, "tokenizer.json"), os.path.join(SRC, "tokenizer_config.json"),
                  os.path.join(SRC, "chat_template.jinja")):
        if os.path.exists(extra):
            dst = os.path.join(STAGE, os.path.basename(extra))
            if not os.path.exists(dst):
                shutil.copy(extra, dst)
    for extra in ("preprocessor_config.json", "video_preprocessor_config.json"):
        src = os.path.join(PROC, extra)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(STAGE, extra))
    json.dump(flat, open(os.path.join(STAGE, "config.json"), "w"), indent=1)
    for w in os.listdir(SRC):
        if w.endswith(".safetensors"):
            dst = os.path.join(STAGE, w)
            if not os.path.exists(dst):
                os.symlink(os.path.join(SRC, w), dst)
    # the vision tower is converted from the same checkpoint, but the mmproj converter wants a VL architecture name
    vcfg = dict(cfg)
    vcfg["architectures"] = ["Qwen3_5ForConditionalGeneration"]
    vcfg["model_type"] = "qwen3_5"
    json.dump(vcfg, open(os.path.join(STAGE, "config.json.vl"), "w"), indent=1)
    print("staged:", STAGE, "| text keys:", len(flat), "| layers:", flat.get("num_hidden_layers"))


def extract_head():
    """score.weight (3 x hidden) -> npz. The GGUF carries the backbone; the decision stays here."""
    import numpy as np
    from safetensors import safe_open
    with safe_open(os.path.join(SRC, "model.safetensors"), framework="pt") as f:
        w = f.get_tensor("score.weight").float().numpy()
    np.savez(HEAD, score_weight=w, labels=["contradiction", "entailment", "neutral"])
    print(f"saved head {w.shape} -> {HEAD}  (norm {np.linalg.norm(w):.3f})")


def register():
    sys.path.insert(0, LLAMA)
    sys.path.insert(1, os.path.join(LLAMA, "gguf-py"))
    from conversion import ModelBase, TEXT_MODEL_MAP, ModelType
    from conversion.qwen import Qwen3_5TextModel

    TEXT_MODEL_MAP[ARCH] = "qwen"

    @ModelBase.register(ARCH)
    class JevQwen35SeqCls(Qwen3_5TextModel):
        """Same as Qwen3_5ForCausalLM, but the classification head is not part of the GGUF (see extract_head)."""
        model_arch = Qwen3_5TextModel.model_arch

        def __init__(self, *args, **kwargs):
            # ModelBase.__init__ calls ModelBase.load_hparams() (not self.), which goes through transformers'
            # AutoConfig: that reads this sequence-classification file as a composite Qwen3.5 config and fills the text
            # fields with its own defaults (hidden 4096, ffn 12288, ctx 32768) that do not match the weights
            # (2560 / 9216 / 262144). Pass the staged flat config.json in as hparams instead.
            cfg = json.loads((Path(args[0]) / "config.json").read_text(encoding="utf-8"))
            kwargs["hparams"] = cfg
            super().__init__(*args, **kwargs)

        def modify_tensors(self, data_torch, name, bid):
            if name.startswith("score."):
                return
            yield from super().modify_tensors(data_torch, name, bid)

    print("registered text class:", ModelBase._model_classes[ModelType.TEXT].get(ARCH))
    print("vision pass uses Qwen3_5ForConditionalGeneration (stock Qwen3VLVisionModel)")


def run_converter(out: str, outtype: str, mmproj: bool, extra: list[str]):
    argv = ["convert_hf_to_gguf.py", STAGE, "--outfile", out, "--outtype", outtype] + extra
    if not mmproj:
        # the checkpoint ships no MTP/nextn tensors (all 32 blocks have weights), so the nextn draft block must not be
        # declared - otherwise the GGUF claims 33 blocks and llama.cpp looks for a block that is not there
        argv.append("--no-mtp")
    if mmproj:
        argv.append("--mmproj")
    sys.argv = argv
    os.chdir(LLAMA)
    runpy.run_path(os.path.join(LLAMA, "convert_hf_to_gguf.py"), run_name="__main__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", action="store_true")
    ap.add_argument("--mmproj", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--outtype", default="f16", choices=["f16", "bf16", "f32", "q8_0"])
    ap.add_argument("--restage", action="store_true")
    ap.add_argument("--extra", nargs="*", default=[])
    args = ap.parse_args()

    stage(force=args.restage)
    if args.text:
        extract_head()
    register()
    if args.mmproj and os.path.exists(os.path.join(STAGE, "config.json.vl")):
        # the mmproj converter reads config.json, so swap in the VL architecture name for the vision pass
        os.replace(os.path.join(STAGE, "config.json"), os.path.join(STAGE, "config.text.json"))
        os.replace(os.path.join(STAGE, "config.json.vl"), os.path.join(STAGE, "config.json"))
    try:
        run_converter(args.out, args.outtype, args.mmproj, list(args.extra))
    finally:
        if args.mmproj and os.path.exists(os.path.join(STAGE, "config.text.json")):
            os.replace(os.path.join(STAGE, "config.json"), os.path.join(STAGE, "config.json.vl"))
            os.replace(os.path.join(STAGE, "config.text.json"), os.path.join(STAGE, "config.json"))


if __name__ == "__main__":
    main()
