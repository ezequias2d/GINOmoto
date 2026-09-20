#!/usr/bin/env python
"""Text-only check of the served jev: three NLI controls (entailment / contradiction / neutral) plus the timing of the
chat-sized batch. Run it after a quantization change: a classifier can shift its probabilities when its weights do."""
from __future__ import annotations

import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from jev.model_llama import JevGGUF  # noqa: E402

CONTROLS = [("A man is playing a guitar.", "Someone is making music.", "entailment"),
            ("A man is playing a guitar.", "The man is asleep in bed.", "contradiction"),
            ("A man is playing a guitar.", "The man is a professional musician.", "neutral")]


def main():
    jev = JevGGUF()
    p = jev.probs("A man is playing a guitar.", [h for _, h, _ in CONTROLS])
    ok = 0
    for (_, h, gold), row in zip(CONTROLS, p):
        got = jev.labels[int(row.argmax())]
        ok += got == gold
        print(f"  {h:38s} -> {got:14s} ({row[0]:.2f}/{row[1]:.2f}/{row[2]:.2f}) gold={gold} {'OK' if got == gold else 'WRONG'}", flush=True)
    print(f"controls: {ok}/3", flush=True)
    t0 = time.perf_counter()
    jev.rerank("Bot commands. The player says: 'pega madeira pra mim'.",
               ["The player is asking the bot to chop wood.", "The player is asking the bot to swim in the ocean.",
                "The player is asking the bot to craft a pickaxe."])
    print(f"rerank of 3 options: {time.perf_counter() - t0:.1f}s", flush=True)
    print(jev.report(), flush=True)


if __name__ == "__main__":
    main()
