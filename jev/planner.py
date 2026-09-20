"""Backward chaining over the Minecraft tech tree, judged by the jev NLI cross-encoder.

From AlexWortega/openjev `code/minecraft.py` (MIT): the tree (recipes) lives in this scaffold, every judgement about the
state is the model's — it is asked a predicate ("The player has at least 4 planks.") and its negation, and the
predicate counts as satisfied when P(entailment) of the positive beats the negative. The first unsatisfied requirement
becomes the next sub-goal; a node whose requirements all hold is executed. Environment feedback corrects it: when an
action fails the state does not change, so the least confident "yes" of that node is flipped.
"""
from __future__ import annotations

import numpy as np

TREE = {
    "log":            {"skill": ("collect", "log"), "batch": 4, "reqs": [("visible", "log")]},
    "planks":         {"skill": ("craft", "planks"), "reqs": [("have", "log", 1)]},
    "stick":          {"skill": ("craft", "stick"), "reqs": [("have", "planks", 2)]},
    "crafting_table": {"skill": ("craft", "crafting_table"), "reqs": [("have", "planks", 4)]},
    "wooden_pickaxe": {"skill": ("craft", "wooden_pickaxe"), "reqs": [("have", "stick", 2), ("have", "planks", 3), ("near", "crafting_table")]},
    "cobblestone":    {"skill": ("collect", "stone"), "reqs": [("have", "wooden_pickaxe", 1), ("visible", "stone")]},
    "stone_pickaxe":  {"skill": ("craft", "stone_pickaxe"), "reqs": [("have", "stick", 2), ("have", "cobblestone", 3), ("near", "crafting_table")]},
    "furnace":        {"skill": ("craft", "furnace"), "reqs": [("have", "cobblestone", 8), ("near", "crafting_table")]},
    "raw_iron":       {"skill": ("collect", "iron_ore"), "reqs": [("have", "stone_pickaxe", 1), ("visible", "iron_ore")]},
    "iron_ingot":     {"skill": ("smelt", "raw_iron"), "reqs": [("have", "raw_iron", 3), ("have", "planks", 2), ("near", "furnace")]},
    "iron_pickaxe":   {"skill": ("craft", "iron_pickaxe"), "reqs": [("have", "iron_ingot", 3), ("have", "stick", 2), ("near", "crafting_table")]},
}
MILESTONES = list(TREE)
NAME = {"log": ("log", "logs"), "planks": ("plank", "planks"), "stick": ("stick", "sticks"), "crafting_table": ("crafting table", "crafting tables"),
        "wooden_pickaxe": ("wooden pickaxe", "wooden pickaxes"), "cobblestone": ("cobblestone block", "cobblestone blocks"),
        "stone_pickaxe": ("stone pickaxe", "stone pickaxes"), "furnace": ("furnace", "furnaces"), "raw_iron": ("raw iron chunk", "raw iron chunks"),
        "iron_ingot": ("iron ingot", "iron ingots"), "iron_pickaxe": ("iron pickaxe", "iron pickaxes"),
        "stone": ("stone", "stone"), "iron_ore": ("iron ore", "iron ore")}
VERB = {"collect": "mine", "craft": "craft", "smelt": "smelt", "place": "place"}
nm = lambda item, n=1: NAME.get(item, (item.replace("_", " "),) * 2)[0 if n == 1 else 1]


def skill_text(sk):
    kind, arg = sk[0], sk[1]
    if kind == "explore":
        return "walk away to explore a new area"
    try:
        if kind == "collect":
            return f"mine {nm(arg)} blocks to get {nm(next(i for i in TREE if TREE[i]['skill'] == (kind, arg)), 2)}"
        if kind == "smelt":
            return f"smelt {nm(arg)} into iron ingots in the furnace"
        if kind in VERB:
            return f"{VERB[kind]} {'' if arg == 'planks' else ('an ' if nm(arg)[0] in 'aeiou' else 'a ')}{nm(arg, 2 if arg == 'planks' else 1)}"
    except (KeyError, IndexError, StopIteration):
        pass
    return f"{kind} {arg or ''}".strip()   # chat skills: follow/come/goto/stop/attack/look/tower/digdown/toss/say


def render_state(s: dict, goal: str | None = None, last: str | None = None, recipes: bool = False) -> str:
    """Explicit zero counts and yes/no sentences: with a sparse "Inventory: 4 logs" the model hallucinates sticks."""
    inv = ", ".join(f"{s['inventory'].get(i, 0)} {nm(i, s['inventory'].get(i, 0))}" for i in TREE)
    near = " ".join(f"A placed {nm(k)} {'stands' if v else 'does not stand'} nearby." for k, v in s["near"].items())
    vis = " ".join(f"{nm(k).capitalize()} blocks {'are' if v else 'are not'} visible nearby." for k, v in s["visible"].items())
    t = "Minecraft survival. "
    if goal:
        t += f"The goal is to craft {'an' if nm(goal)[0] in 'aeiou' else 'a'} {nm(goal)}. "
    if recipes:
        t += ("Recipes: planks = 1 log; sticks = 2 planks; crafting table = 4 planks; wooden pickaxe = 3 planks + 2 sticks; stone pickaxe = 3 cobblestone + 2 sticks; "
              "furnace = 8 cobblestone; iron pickaxe = 3 iron ingots + 2 sticks. Pickaxes and the furnace are crafted at a placed crafting table. Stone needs a wooden pickaxe, "
              "iron ore needs a stone pickaxe, iron ingots are smelted from raw iron in a placed furnace with planks as fuel. ")
    t += f"Inventory counts: {inv}. {near} {vis}"
    if last:
        t += f" Last action: {last}"
    return t


def req_hyps(r, negated: bool = False, phrasings: int = 3) -> list[str]:
    """A predicate (or its negation) in one or several phrasings. The planner sums P(entailment) over the phrasings."""
    if r[0] == "have":
        i, n = r[1], r[2]
        out = [f"The player has {'fewer than' if negated else 'at least'} {n} {nm(i, n)}.",
               f"The {nm(i)} count in the inventory is {'smaller than' if negated else 'greater than or equal to'} {n}.",
               f"The number of {nm(i, 2)} in the inventory is {'less than ' + str(n) if negated else str(n) + ' or more'}."]
        return out[:phrasings]
    if r[0] == "near":
        return [f"The answer to whether a {nm(r[1])} is placed nearby is {'no' if negated else 'yes'}."]
    return [f"{'No' if negated else 'A'} {nm(r[1])} block is visible nearby."]


req_hyp = lambda r: req_hyps(r)[0]


def req_truth(r, s: dict) -> bool:
    return s["inventory"].get(r[1], 0) >= r[2] if r[0] == "have" else (s["near"][r[1]] if r[0] == "near" else s["visible"][r[1]])


def make_judge(jev, phrasings: int = 3, rule: str = "pair", trace: list | None = None):
    """`judge(state, reqs) -> margins`, margins > 0 = requirement satisfied."""
    def judge(s, reqs):
        hyps, owner = [], []
        for k, r in enumerate(reqs):
            for sign, neg in ((1, False), (-1, True)):
                hs = req_hyps(r, neg, phrasings)
                for h in hs:
                    hyps.append(h)
                    owner.append((k, sign / len(hs)))
        p = jev.probs(render_state(s), hyps)[:, 1]
        pos, neg = np.zeros(len(reqs)), np.zeros(len(reqs))
        for (k, w), pi in zip(owner, p):
            (pos if w > 0 else neg)[k] += abs(w) * pi
        margin = [float(a - b) if rule == "pair" else float(a - 0.5) for a, b in zip(pos, neg)]
        if trace is not None:
            for r, a, b, mg in zip(reqs, pos, neg, margin):
                trace.append({"hyp": req_hyp(r), "p_ent": float(a), "p_ent_neg": float(b), "judged": mg > 0, "truth": bool(req_truth(r, s))})
        return margin
    return judge


class Chain:
    """Backward chaining; `judge` is the model (or ground truth). Same algorithm as the repo's planner."""

    def __init__(self, goal: str, judge):
        self.goal, self.judge, self.fails = goal, judge, {}

    def decide(self, s, trace, failed=()):
        self.fails = {k: v for k, v in self.fails.items() if k in failed}
        for sk in failed:
            self.fails.setdefault(sk, 0)
        trace.append({"node": "goal reached?"})
        if self.judge(s, [("have", self.goal, 1)])[0] > 0:
            return None
        item, n = self.goal, 1
        for _ in range(12):
            trace.append({"node": item})
            reqs = TREE[item]["reqs"]
            m = list(self.judge(s, reqs))
            sk = TREE[item]["skill"]
            sk = sk + (max(n, TREE[item].get("batch", 1)) if sk[0] == "collect" else 1,)
            if all(x > 0 for x in m) and sk in failed:
                self.fails[sk] += 1
                for k in np.argsort(m)[:self.fails[sk]]:
                    m[k] = -1.0
                    trace.append({"node": f"action failed -> doubting: {req_hyp(reqs[k])}"})
            miss = next((r for r, x in zip(reqs, m) if x <= 0), None)
            if miss is None:
                return sk
            if miss[0] == "visible":
                return ("explore", None, 1)
            if miss[0] == "near":
                trace.append({"node": "placed " + miss[1]})
                if self.judge(s, [("have", miss[1], 1)])[0] > 0:
                    return ("place", miss[1], 1)
                item, n = miss[1], 1
            else:
                item, n = miss[1], miss[2]
        return ("explore", None, 1)
