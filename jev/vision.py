"""Pixels for the jev model.

The symbolic state the skill server returns is already text, so a text-only jev is enough for the tech tree. These are
the questions it *cannot* answer from that state, and which need the frame: is there a tree in front, is the bot about
to walk into the ocean, is it in a cave. Frames come from the prismarine-viewer through bot/shot.js (headless Chrome),
and the model answers with entailment probabilities over a fixed set of statements about the picture.
"""
from __future__ import annotations

import os
import time

# one scene statement per hypothesis, plus how the explore policy should steer (degrees, 0 = keep walking forward)
PREMISE = ("A first-person Minecraft screenshot taken by a bot standing in a survival world, the crosshair at the exact "
           "centre of the image: {img} Add nothing to the picture.")
SCENE = [
    ("A tree trunk with leaves is standing right in front of the bot.", "tree", 0),
    ("The bot is facing water: a lake, river or the ocean fills the middle of the view.", "water", 180),
    ("The bot is inside a cave, a tunnel or a dark underground room.", "cave", 180),
    ("The view is a bare stone cliff, a rock face or a mountain side with no trees.", "stone", 135),
    ("The bot is looking out over open flat grass, with nothing but sky above the horizon.", "open", 45),
    ("The picture is mostly sky: the bot is looking up or standing on a high ledge.", "sky", 90),
]


class Vision:
    """Answers 'what is on screen' with the same entailment primitive the planner uses."""

    def __init__(self, jev, shots_dir: str = "shots", tag: str = "view"):
        self.jev, self.shots_dir, self.tag, self.n = jev, shots_dir, tag, 0
        os.makedirs(shots_dir, exist_ok=True)
        self.log = []

    def look(self, frame_path: str) -> dict:
        t0 = time.perf_counter()
        p = self.jev.probs_img(frame_path, PREMISE, [h for h, _, _ in SCENE])
        rows = [{"statement": h, "tag": tag, "steer": steer,
                 "p_ent": float(p[i, 1]), "p_con": float(p[i, 0]), "p_neu": float(p[i, 2])}
                for i, (h, tag, steer) in enumerate(SCENE)]
        best = max(rows, key=lambda r: r["p_ent"])
        out = {"frame": frame_path, "verdict": best["statement"], "tag": best["tag"], "steer": best["steer"],
               "p_ent": best["p_ent"], "margin": best["p_ent"] - sorted(r["p_ent"] for r in rows)[-2],
               "rows": rows, "secs": time.perf_counter() - t0, "ms": (time.perf_counter() - t0) * 1000}
        self.log.append(out)
        return out
