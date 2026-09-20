"""Chat orders: the player talks to the bot in game, the jev model decides what he asked for.

The cross-encoder cannot generate a command parser, but it can score statements about the order, so each intent is one
English statement about what the player wants and the intent with the highest P(entailment) is executed. Targets
(which item, which goal) come from the same primitive over a closed candidate list. Messages are usually Portuguese —
the model was fine-tuned on English NLI — so every interpretation is scored, the score is reported, and a keyword
fallback catches the cases where the entailment ranking is not confident (see `interpret`).
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np

from jev.model import ENT

PREMISE = ('A player is giving orders to his Minecraft bot in chat, in Portuguese. He writes: "{msg}"\n'
           "What he wants the bot to do is this:")

INTENTS = [
    ("mine_logs",  "The player wants the bot to chop down trees and collect wood.",              "collect:log:4"),
    ("mine_stone", "The player wants the bot to mine stone and collect cobblestone.",            "collect:stone:4"),
    ("mine_iron",  "The player wants the bot to mine iron ore.",                                 "collect:iron_ore:3"),
    ("craft",      "The player wants the bot to craft an item for him.",                          "craft:target"),
    ("goal",       "The player wants the bot to get or craft a specific item, whatever it takes.", "goal:target"),
    ("follow",     "The player wants the bot to follow him around.",                              "follow:"),
    ("come",       "The player wants the bot to walk to him and stand next to him.",               "come:"),
    ("goto",       "The player gives the bot coordinates in the world to walk to.",                "goto:"),
    ("stop",       "The player wants the bot to stop what it is doing and wait.",                  "stop:"),
    ("attack",     "The player wants the bot to fight the monsters nearby.",                       "attack:"),
    ("tower",      "The player wants the bot to build a tower out of blocks and climb up it.",     "tower:3"),
    ("dig_down",   "The player wants the bot to dig straight down into the ground.",              "digdown:3"),
    ("give",       "The player wants the bot to hand over or drop items for him.",                 "toss:target"),
    ("look",       "The player wants the bot to look at him.",                                     "look:"),
    ("status",     "The player asks the bot what it is doing and what it has in its inventory.",   "status:"),
    ("explore",    "The player wants the bot to wander off and explore the area.",                 "explore:"),
    ("greet",      "The player greets the bot or asks how it is doing.",                          "reply:greet"),
    ("praise",     "The player is praising the bot or telling it that it did well.",               "reply:praise"),
    ("quiet",      "The player tells the bot to talk less or to shut up.",                        "quiet:"),
]

# target candidates: (phrase the model scores, what the scaffold calls it)
TARGETS = [
    ("logs of wood", "log"), ("wooden planks", "planks"), ("sticks", "stick"), ("a crafting table", "crafting_table"),
    ("cobblestone", "cobblestone"), ("a wooden pickaxe", "wooden_pickaxe"), ("a stone pickaxe", "stone_pickaxe"),
    ("iron ingots", "iron_ingot"), ("raw iron chunks", "raw_iron"), ("an iron pickaxe", "iron_pickaxe"),
    ("stone blocks", "stone"), ("iron ore", "iron_ore"), ("a furnace", "furnace"),
]

# only used when the entailment ranking is not confident: order matters, first match wins
KEYWORDS = [
    (r"\b(para|pare|quieto|quieta|espera|fica ai|fica a[ií]|stop|wait|calma)\b", "stop"),
    (r"\b(segue|me segue|follow|vem comigo|cola em mim|atras de mim|atrás de mim)\b", "follow"),
    (r"\b(vem|venha|chega|aproxima|aproxima|come|encosta)\b", "come"),
    (r"\b(madeira|tronco|lenha|arvore|árvore|wood|log|chop)\b", "mine_logs"),
    (r"\b(ferro|iron ore|minerio de ferro|minério de ferro|iron)\b", "mine_iron"),
    (r"\b(pedra|stone|cascalho|cobble|pedregulho)\b", "mine_stone"),
    (r"\b(mata|matar|mate|zumbi|zombie|monstro|mob|inimigo|luta|briga|defende|kill|attack)\b", "attack"),
    (r"\b(torre|tower|pilar|sobe|subir|escalar)\b", "tower"),
    (r"\b(cava pra baixo|cavar pra baixo|escava pra baixo|pra baixo|pra baixo|dig down|cava aqui)\b", "dig_down"),
    (r"\b(me da|me dá|dropa|dropa|joga|larga|passa|da pra mim|dá pra mim|entrega)\b", "give"),
    (r"\b(olha|observa|look|encara)\b", "look"),
    (r"\b(inventario|inventário|o que voce|o que você|status|como voce ta|como você tá|ta fazendo o que|tá fazendo o quê)\b", "status"),
    (r"\b(crafta|craftar|faz|fazer|constroi|construi|monta|preciso de|quero)\b", "craft"),
    (r"\b(explora|explorar|passeia|vagar|vai por ai|vai por aí|explore|anda)\b", "explore"),
    (r"\b(cala|calado|silencio|silêncio|chega de falar|fala menos|quiet)\b", "quiet"),
    (r"\b(oi|ola|olá|eae|e ai|e aí|beleza|tudo bem|bom dia|boa tarde|boa noite|hey|hi|hello)\b", "greet"),
    (r"\b(muito bem|mandou bem|boa|isso|top|valeu|obrigado|obrigada|brigado|nice|well done)\b", "praise"),
]


def norm(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")


def extract_coords(msg: str):
    nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", msg.replace(",", " "))]
    return nums if len(nums) >= 2 else None


ITEM_HINTS = {  # pt-BR word -> candidate targets (the model picks among them, it does not have to scan all 13)
    "madeira": ["log", "planks"], "tronco": ["log", "planks"], "lenha": ["log", "planks"], "arvore": ["log"],
    "tabua": ["planks"], "graveto": ["stick"], "palito": ["stick"], "pedra": ["cobblestone", "stone"],
    "pedregulho": ["cobblestone"], "ferro": ["iron_ore", "raw_iron", "iron_ingot"], "picareta": ["wooden_pickaxe", "stone_pickaxe", "iron_pickaxe"],
    "bancada": ["crafting_table"], "mesa": ["crafting_table"], "fornalha": ["furnace"], "bau": ["crafting_table"],
    "minerio": ["iron_ore", "raw_iron"],
}


def keyword_targets(msg: str) -> list[str]:
    m = norm(msg)
    out = []
    for word, items in ITEM_HINTS.items():
        if re.search(rf"\b{word}", m):
            out += [i for i in items if i not in out]
    return out


def keyword_intent(msg: str):
    m = norm(msg)
    for pat, name in KEYWORDS:
        if re.search(pat, m):
            return name
    return None


class Chat:
    """Interprets player messages with the jev model, with a keyword fallback and a report of both."""

    def __init__(self, jev, min_conf: float = 0.55, min_margin: float = 0.05, verbose: bool = True, shorten: bool = True):
        self.jev, self.min_conf, self.min_margin, self.verbose = jev, min_conf, min_margin, verbose
        self.shorten = shorten  # score only the candidates the wording points at (see interpret)
        self.hyps = [h for _, h, _ in INTENTS]
        self.log = []

    def interpret(self, msg: str) -> dict:
        premise = PREMISE.format(msg=msg)
        kw = keyword_intent(msg)
        # Every hypothesis is its own sequence through the model, and on this machine a sequence costs ~3.6 s, so the
        # 19 intents are only scored when the keyword pass finds nothing (or when --full-intents is on). With a hit the
        # model still decides — but among the candidates the wording pointed at, plus a "none of these" option.
        cand = [i for i, (n, _, _) in enumerate(INTENTS) if n == kw] if kw else []
        extra = [i for i, (n, _, _) in enumerate(INTENTS) if n not in (kw, "quiet")][:2]
        idx = cand + [i for i in extra if i not in cand] if cand else list(range(len(INTENTS)))
        if self.shorten and cand:
            idx = cand + extra
        p_full = self.jev.probs(premise, [self.hyps[i] for i in idx])[:, ENT]
        p = np.zeros(len(self.hyps), dtype=np.float32)
        p[idx] = p_full
        order = p.argsort()[::-1]
        top, second = INTENTS[order[0]], p[order[1]]
        conf, margin = float(p[order[0]]), float(p[order[0]] - second)
        kw = keyword_intent(msg)
        used = "jev"
        name = top[0]
        action = top[2]
        if conf < self.min_conf or margin < self.min_margin:
            if kw:
                name, action, used = kw, next(a for n, _, a in INTENTS if n == kw), "keyword"
        target = None
        if "target" in action:
            hinted = keyword_targets(msg)
            cand = [t for t in TARGETS if t[1] in hinted] if (self.shorten and hinted) else TARGETS
            tp = self.jev.probs(PREMISE.format(msg=msg), [f"The player is asking for {t}." for t, _ in cand])[:, ENT]
            target = cand[int(tp.argmax())][1]
            action = action.replace("target", target)
        if kw and kw != name and self.verbose:
            print(f"      [chat] jev says {top[0]} ({conf:.2f}, margin {margin:.2f}) / keywords say {kw} -> {name}", flush=True)
        out = {"msg": msg, "intent": name, "action": action, "conf": conf, "margin": margin,
               "used": used, "target": target, "keyword": kw, "ranked": [(INTENTS[i][0], float(p[i])) for i in order[:4]]}
        self.log.append(out)
        return out
