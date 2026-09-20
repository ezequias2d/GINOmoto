#!/usr/bin/env python
"""Does the image actually reach the NLI head? Same premise+hypotheses, two very different frames: the entailment table
should differ, and the hidden state of the last token must not be identical."""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from jev.model_llama import JevGGUF  # noqa: E402
from jev.vision import PREMISE, SCENE  # noqa: E402

FRAMES = sys.argv[1:] or ["shots/check.png", "/tmp/visiontest.png"]


def main():
    jev = JevGGUF()
    print("media marker:", jev.marker)
    vecs = {}
    for f in FRAMES:
        t0 = time.perf_counter()
        p = jev.probs_img(f, PREMISE, [h for h, _, _ in SCENE])
        dt = time.perf_counter() - t0
        print(f"\n{f}  ({dt:.1f}s, {len(SCENE)} hypotheses)")
        for (h, tag, steer), row in zip(SCENE, p):
            print(f"   p_ent={row[1]:.2f} p_con={row[0]:.2f} p_neu={row[2]:.2f}  [{tag:5s}] {h}")
        best = int(np.argmax(p[:, 1]))
        print(f"   verdict: {SCENE[best][0]}  (p_ent {p[best, 1]:.2f})")
        vecs[f] = p
    if len(FRAMES) > 1:
        a, b = vecs[FRAMES[0]][:, 1], vecs[FRAMES[1]][:, 1]
        print(f"\nentailment profiles differ? {not np.allclose(a, b, atol=1e-3)}  "
              f"(mean |diff| {np.abs(a - b).mean():.3f}, cosine {np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9):.3f})")


if __name__ == "__main__":
    main()
