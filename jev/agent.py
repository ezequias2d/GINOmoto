#!/usr/bin/env python
"""jev plays real Minecraft — and takes orders from the player in chat.

Two ways to use the model, both of them an entailment check:
  * the tech-tree scaffold (jev/planner.py): "the player has at least 2 sticks" vs its negation, backward chained, and
    the skill server (mineflayer, bot/skill_server.js) executes the action in a real 1.20.4 world;
  * the chat layer (jev/chat.py): every message the player types becomes a premise, each intent is a statement about
    what he wants ("The player wants the bot to follow him around."), and the argmax entailment is the order the bot
    obeys. Targets (which item) are chosen from a closed candidate list the same way.
When the scaffold runs out of visible resources the bot looks at its own screen (prismarine-viewer -> headless Chrome ->
jev/vision.py) and the same primitive says what is in front of it, which steers the walk.

    python -m jev.agent --goal iron_pickaxe --vision            # chat mode: obey the player, otherwise keep the goal
    python -m jev.agent --episodes 1 --vision --max-steps 25    # benchmark mode: N scaffold episodes, no chat
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jev.backend import make_jev                   # noqa: E402
from jev.chat import Chat                          # noqa: E402
from jev.mc import SkillBot, shot                  # noqa: E402
from jev.model import ENT                          # noqa: E402
from jev.planner import Chain, MILESTONES, make_judge, skill_text, TREE  # noqa: E402
from jev.vision import Vision                      # noqa: E402

SPEECH = {
    "start": "Bora. Meta: {goal}.",
    "milestone": "Consegui: {item}.",
    "goal": "Meta batida: {goal}.",
    "tree": "Tem árvore na minha frente, indo pra cima.",
    "water": "Água na frente — meia-volta.",
    "cave": "Tô no escuro, voltando pra fora.",
    "stone": "Parede de pedra na cara, virando.",
    "sky": "Só céu, vou descer daqui.",
    "open": "Campo aberto, seguindo.",
    "stuck": "Travei nesse passo, mudando de plano.",
    # replies to chat orders
    "ack": {
        "collect": "Beleza {who}, vou pegar {what}.",
        "craft": "Vou craftar {what}, {who}.",
        "goal": "Fechado {who}: {what}. Aguenta.",
        "follow": "Tô colado em você, {who}.",
        "come": "Já vou, {who}.",
        "goto": "Indo pra {what}.",
        "stop": "Parei. Manda.",
        "attack": "Já era, {who}. Partindo pra cima.",
        "tower": "Subindo a torre.",
        "digdown": "Cavando pra baixo, cuidado com o buraco.",
        "give": "Toma: {what}.",
        "look": "Olhando pra você, {who}.",
        "status": "Tô com {what}.",
        "explore": "Vou dar uma volta.",
        "greet": "Eae {who}. Manda o serviço.",
        "praise": "Sempre, {who}.",
        "quiet": "Fechado, fico na minha.",
        "unquiet": "Voltei a falar.",
        "unknown": "Não entendi essa, {who}. Repete no miúdo.",
    },
    "busy": "Tô no meio de uma parada, {who} — fala de novo em uns segundos.",
}

URGENT = {"stop", "come", "follow", "attack", "look"}


# ----------------------------------------------------------------------------- dashboard
class Dash:
    def __init__(self, port: int, shot_url: str = "http://127.0.0.1:3008/shot.png"):
        self.port, self.shot_url, self.state = port, shot_url, {"status": "starting"}

    def serve(self):
        dash = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.startswith("/api/state"):
                    body = json.dumps(dash.state, default=str).encode()
                    self.send_response(200)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(body)))
                    self.end_headers()
                    return self.wfile.write(body)
                if self.path.startswith("/shot.png"):
                    try:
                        import urllib.request
                        b = urllib.request.urlopen(dash.shot_url, timeout=20).read()
                    except Exception:
                        return self.send_response(204)
                    self.send_response(200)
                    self.send_header("content-type", "image/png")
                    self.send_header("cache-control", "no-store")
                    self.send_header("content-length", str(len(b)))
                    self.end_headers()
                    return self.wfile.write(b)
                html = dash.page().encode()
                self.send_response(200)
                self.send_header("content-type", "text/html; charset=utf-8")
                self.send_header("content-length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)

        ThreadingHTTPServer.allow_reuse_address = True
        srv = ThreadingHTTPServer(("127.0.0.1", self.port), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv

    @staticmethod
    def page() -> str:
        return """<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><title>jev · minecraft</title>
<style>
 body{background:#0d1117;color:#e6edf3;font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;margin:0;padding:18px}
 h1{font-size:15px;letter-spacing:.14em;text-transform:uppercase;color:#7ee787;margin:0 0 14px}
 .row{display:flex;gap:16px;flex-wrap:wrap}
 .card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px 14px;min-width:300px;flex:1}
 .card h2{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:#8b949e;margin:0 0 8px}
 img{width:100%;border-radius:8px;border:1px solid #30363d;background:#000}
 .bar{height:7px;background:#21262d;border-radius:4px;overflow:hidden;margin:2px 0 7px}
 .bar i{display:block;height:100%;background:linear-gradient(90deg,#238636,#7ee787)}
 .dim{color:#8b949e}.ok{color:#7ee787}.bad{color:#ff7b72}.hi{color:#79c0ff}.who{color:#d2a8ff}
 .st{display:flex;justify-content:space-between;gap:10px}
</style></head><body>
<h1>jev · minecraft bot</h1>
<div class="row">
 <div class="card" style="flex:1.3"><h2>visor do bot (prismarine-viewer)</h2><img id="shot" src="/shot.png"><p class="dim" id="cam"></p></div>
 <div class="card"><h2>estado</h2><div id="st"></div></div>
 <div class="card"><h2>chat</h2><div id="ch"></div></div>
</div>
<div class="row" style="margin-top:16px">
 <div class="card"><h2>decisões do jev (P entailment)</h2><div id="tr"></div></div>
 <div class="card"><h2>o que ele vê</h2><div id="vis"></div></div>
</div>
<script>
const J = o => JSON.stringify(o);
async function tick(){
 try{
  const s = await (await fetch('/api/state')).json();
  st.innerHTML = `<div class="st"><span class="dim">status</span><b class="${s.status==='running'?'ok':'bad'}">${s.status||''}</b></div>`
   + `<div class="st"><span class="dim">meta</span><b class="hi">${s.goal||''}</b></div>`
   + `<div class="st"><span class="dim">ação</span><b>${s.action||''}</b></div>`
   + `<div class="st"><span class="dim">resultado</span><b class="${s.ok?'ok':'bad'}">${s.msg||''}</b></div>`
   + `<div class="st"><span class="dim">inventário</span><b>${J(s.inventory||{})}</b></div>`
   + `<div class="st"><span class="dim">visível</span><b>${J(s.visible||{})}</b></div>`
   + `<div class="st"><span class="dim">jogadores</span><b>${J(s.players||{})}</b></div>`
   + `<div class="st"><span class="dim">modelo</span><b>${s.calls||0} calls · ${s.ms_per_call||0} ms/call</b></div>`;
  ch.innerHTML = (s.chat||[]).map(m => `<div><span class="who">${m.username}</span>: ${m.message}<br><span class="dim">${m.intent||''} ${m.conf!==undefined?m.conf.toFixed(2):''} ${m.used||''} ${m.reply?('→ '+m.reply):''}</span></div>`).join('') || '<span class="dim">—</span>';
  tr.innerHTML = (s.trace||[]).map(t => t.hyp
     ? `<div>${t.hyp} <span class="${t.judged?'ok':'bad'}">${t.p_ent.toFixed(2)}${t.truth!==undefined?` / real ${t.truth}`:''}</span></div><div class="bar"><i style="width:${(t.p_ent*100).toFixed(0)}%"></i></div>`
     : `<div class="dim">→ ${t.node}</div>`).join('');
  vis.innerHTML = (s.vision||[]).map(v => `<div>${v.verdict} <span class="hi">${v.p_ent.toFixed(2)}</span> <span class="dim">${v.steer}° ${v.ms.toFixed(0)} ms</span></div>`).join('') || '<span class="dim">—</span>';
  cam.textContent = s.frame ? s.frame.split('/').pop() : '';
 }catch(e){}
}
setInterval(tick, 1500); tick();
setInterval(()=>{document.getElementById('shot').src='/shot.png?t='+Date.now()}, 3000);
</script></body></html>"""


# ----------------------------------------------------------------------------- skill worker
class Worker(threading.Thread):
    """Runs one blocking skill at a time; the main loop keeps reading chat while this is busy."""

    def __init__(self, bot):
        super().__init__(daemon=True)
        self.bot, self.q = bot, queue.Queue()
        self.busy, self.last, self.result = False, None, None

    def submit(self, sk):
        self.busy, self.result, self.last = True, None, sk
        self.q.put(sk)

    def run(self):
        while True:
            sk = self.q.get()
            r = {"ok": False, "msg": "error"}
            try:
                r = self.bot.act(sk)
            except Exception as e:
                r = {"ok": False, "msg": f"exception: {e}"}
            self.result, self.busy = r, False

    def wait(self, timeout=None):
        t0 = time.time()
        while self.busy and (timeout is None or time.time() - t0 < timeout):
            time.sleep(0.4)
        return self.result


# ----------------------------------------------------------------------------- agent
class Agent:
    def __init__(self, args):
        self.args = args
        self.said = []                      # our own recent lines: other bots on the server echo them back as chat
        self.ignore = {n.strip() for n in (args.ignore or "").split(",") if n.strip()}
        self.ignore_prefixes = tuple(p.strip().lower() for p in (args.ignore_prefix or "").split(",") if p.strip())
        self.dash = Dash(args.dashboard)
        self.bot = SkillBot(args.skill_url)
        self.worker = Worker(self.bot)
        self.chatlog = []

    def boot(self):
        self.dash.serve()
        print(f"dashboard: http://127.0.0.1:{self.args.dashboard}", flush=True)
        self.bot.wait_ready()
        print("skill server ready", flush=True)
        self.jev = make_jev(self.args.backend, url=self.args.server_url, head=self.args.head, ckpt=self.args.ckpt,
                            img_processor=self.args.img_processor, threads=self.args.threads, bs=self.args.bs,
                            image_size=self.args.image_size, api_key=self.args.api_key)
        print(f"jev loaded in {self.jev.load_s:.1f}s (vision tower patched: {self.jev.patch_embed_patched})", flush=True)
        self.vision = Vision(self.jev, self.args.shots)
        self.chat = Chat(self.jev, min_conf=self.args.min_conf, min_margin=self.args.min_margin,
                         shorten=not self.args.full_intents)
        self.worker.start()
        self.bot_name = (self.bot.state() or {}).get("bot", "GINOmoto")
        self.bot.chat(0)  # drop anything said before we started listening
        self.say(SPEECH["start"].format(goal=self.args.goal.replace("_", " ")))

    # ------------------------------------------------------------------ talk
    def say(self, text):
        try:
            self.worker_q_talk(text)
        except Exception as e:
            print("      [chat] failed:", e, flush=True)

    def worker_q_talk(self, text):
        # talking must work while a skill is running: the skill server lets `say` bypass its busy lock
        def fire():
            try:
                self.bot.say(text)
                self.remember(text)
                print(f"      [chat] {text}", flush=True)
            except Exception as e:
                print("      [chat] failed:", e, flush=True)
        threading.Thread(target=fire, daemon=True).start()

    def who(self, msg):
        return msg.get("username", "chefe")

    def reply(self, kind, who, what=""):
        line = SPEECH["ack"].get(kind, SPEECH["ack"]["unknown"]).format(who=who, what=what)
        self.say(line)
        return line

    def remember(self, line):
        self.said.append(line)
        del self.said[:-8]

    # ------------------------------------------------------------------ chat
    def poll_chat(self):
        out = []
        try:
            msgs = self.bot.chat()
        except Exception as e:
            print("      [chat] poll failed:", e, flush=True)
            return out
        for m in msgs:
            text = (m.get("message") or "").strip()
            if not text or text.startswith("/") or text == "!":
                continue
            who = self.who(m)
            low = text.lower()
            if who in self.ignore or who == self.bot_name:
                continue
            # servers often run another chatbot that answers "got it: <what you said>" — feeding that back would make
            # the two bots talk to each other forever
            if any(low.startswith(pref) for pref in self.ignore_prefixes) or any(s and s.lower() in low for s in self.said):
                print(f"      [ordem] ignorando eco de {who}: {text[:60]!r}", flush=True)
                continue
            r = self.chat.interpret(text)
            print(f"      [ordem] {who}: {text!r} -> {r['intent']} ({r['used']}, conf {r['conf']:.2f}, "
                  f"margem {r['margin']:.2f}, alvo {r['target'] or '-'})", flush=True)
            r["username"], r["reply"] = who, ""
            if self.handle(r):
                self.chatlog.append(r)
                out.append(r)
        self.dash.state["chat"] = self.chatlog[-12:]
        return out

    def handle(self, r):
        """Execute one interpreted order. Returns True if it produced a visible reaction."""
        who, action = r["username"], r["action"]
        kind, _, arg = action.partition(":")
        if kind == "reply":
            r["reply"] = self.reply(arg, who)
            return True
        if kind == "quiet":
            self.args.quiet = not self.args.quiet
            r["reply"] = self.reply("quiet" if self.args.quiet else "unquiet", who)
            return True
        if kind == "status":
            s = self.bot.state()
            inv = ", ".join(f"{v} {k}" for k, v in s["inventory"].items() if v) or "nada"
            r["reply"] = self.reply("status", who, inv)
            return True
        if kind == "goto":
            from jev.chat import extract_coords
            coords = [float(x) for x in arg.split()] if arg.strip() else extract_coords(r["msg"])
            if not coords:
                r["reply"] = self.reply("unknown", who)
                return True
            arg = " ".join(str(c) for c in coords)
        if kind == "collect":
            what, n = arg.split(":")
            self.worker.submit(("collect", what, int(n)))
            r["reply"] = self.reply("collect", who, what.replace("_", " "))
            return True
        if kind == "goal":
            self.set_goal(arg)
            r["reply"] = self.reply("goal", who, arg.replace("_", " "))
            return True
        if kind == "craft":
            if self.worker.busy:
                # crafting a single item while the scaffold is working would fight over the inventory: run the goal
                self.set_goal(arg)
                r["reply"] = self.reply("goal", who, arg.replace("_", " "))
                return True
            self.worker.submit(("craft", arg, 1))
            r["reply"] = self.reply("craft", who, arg.replace("_", " "))
            return True
        import re
        nums = [int(n) for n in re.findall(r"\b(\d{1,2})\b", r["msg"])]
        n = nums[0] if nums else 3
        skill = {"follow": ("follow", who, 1), "come": ("come", who, 1), "stop": ("stop", None, 1),
                 "attack": ("attack", None, 1), "tower": ("tower", str(n), n), "digdown": ("digdown", str(n), n),
                 "give": ("toss", r["target"], n), "look": ("look", who, 1), "explore": ("explore", None, 1)}.get(kind)
        if not skill:
            r["reply"] = self.reply("unknown", who)
            return True
        if skill[0] in ("stop", "come", "follow", "attack", "look") and self.worker.busy:
            self.interrupt()
        self.worker.submit(skill)
        r["reply"] = self.reply(kind, who, (r["target"] or "").replace("_", " "))
        return True

    def interrupt(self):
        """A separate call: the skill server lets `stop` cut in while another skill is running."""
        try:
            self.bot.act(("stop", None, 1))
            self.worker.wait(timeout=20)
            print("      [ordem] interrompi o que estava fazendo", flush=True)
        except Exception as e:
            print("      [ordem] interrupt failed:", e, flush=True)

    def set_goal(self, goal):
        if goal not in TREE:
            return
        self.args.goal = goal
        self.chain = Chain(goal, make_judge(self.jev, self.args.phrasings, self.args.rule))
        self.failed = set()

    # ------------------------------------------------------------------ vision
    def look(self, s):
        path = os.path.join(self.args.shots, f"view-{self.vision.n:03d}.png")
        try:
            shot(path, self.args.shot_url)
        except Exception as e:
            print("      [olho] sem frame:", e, flush=True)
            return None
        v = self.vision.look(path)
        self.dash.state["frame"] = path
        print(f"      [olho] {v['tag']:6s} p_ent={v['p_ent']:.2f} margem={v['margin']:.2f} ({v['ms']:.0f} ms) | {v['verdict']}", flush=True)
        self.dash.state["vision"] = [{k: v[k] for k in ("tag", "verdict", "p_ent", "margin", "ms", "frame")}]
        return v

    # ------------------------------------------------------------------ scaffold
    def goal_step(self, s):
        """One backward-chained decision, using the screen when nothing is in sight."""
        trace = []
        self.chain.judge = make_judge(self.jev, self.args.phrasings, self.args.rule, trace)
        t0 = time.time()
        sk = self.chain.decide(s, trace, self.failed)
        ms = (time.time() - t0) * 1000
        if sk is None:
            return None, trace, ms
        if sk[0] == "explore" and self.args.vision:
            v = self.look(s)
            if v:
                if not self.args.quiet and v["tag"] in SPEECH:
                    self.say(SPEECH[v["tag"]])
                sk = ("explore", v["steer"], 1)
        return sk, trace, ms

    # ------------------------------------------------------------------ main
    def update_dash(self, s, action="", ok=None, msg="", trace=None, goal=None):
        self.dash.state.update({"status": "running", "goal": goal or self.args.goal, "action": action, "ok": ok, "msg": msg,
                                "inventory": s.get("inventory", {}), "visible": s.get("visible", {}),
                                "near": s.get("near", {}), "players": s.get("players", {}), "trace": trace or [],
                                "calls": self.jev.calls, "ms_per_call": round(1000 * self.jev.secs / max(1, self.jev.calls))})

    def run_forever(self):
        while True:
            try:
                self._run_forever_once()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"      [erro] laço principal: {type(e).__name__}: {e} — seguindo", flush=True)
                self.dash.state.update({"status": "recuperando", "msg": f"{type(e).__name__}: {e}"})
                self.worker.busy = False
                time.sleep(2)

    def _run_forever_once(self):
        s = self.bot.state()
        self.update_dash(s, action="acordando", msg="")
        reached = {i for i in MILESTONES if s["inventory"].get(i, 0) > 0}
        while True:
            self.poll_chat()
            if self.worker.busy:
                self.update_dash(s, action=skill_text(self.worker.last) if self.worker.last else "", msg="executando")
                time.sleep(1.0)
                continue
            if self.worker.result is not None and self.worker.last is not None:
                r, sk = self.worker.result, self.worker.last
                self.worker.result = None
                print(f"      -> {'ok  ' if r['ok'] else 'FAIL'} {r['msg']}", flush=True)
                s = r.get("state") or self.bot.state()
                new = {i for i in MILESTONES if s["inventory"].get(i, 0) > 0} - reached
                for i in sorted(new, key=MILESTONES.index):
                    if not self.args.quiet:
                        self.say(SPEECH["milestone"].format(item=i.replace("_", " ")))
                reached |= new
                if s["inventory"].get(self.args.goal, 0) > 0:
                    if not self.args.quiet:
                        self.say(SPEECH["goal"].format(goal=self.args.goal.replace("_", " ")))
                    self.args.goal = self.args.after if self.args.after else self.args.goal
                    self.set_goal(self.args.goal)
                self.update_dash(s, action=skill_text(sk) if sk else "", ok=r["ok"], msg=r["msg"])
                continue
            if self.args.chat_only:
                time.sleep(1.5)
                continue
            sk, trace, ms = self.goal_step(s)
            if sk is None:
                self.update_dash(s, action="meta batida", ok=True, msg=f"{self.args.goal} pronto", trace=trace)
                print(f"      == meta {self.args.goal} batida, aguardando ordens", flush=True)
                time.sleep(2.0)
                continue
            print(f"      [plano] {skill_text(sk):52s} ({ms / 1000:.1f}s de modelo)", flush=True)
            self.update_dash(s, action=skill_text(sk), msg="decidindo", trace=trace)
            self.worker.submit(sk)
            self.worker.wait(timeout=1)  # give it a moment so the dashboard shows something

    # ------------------------------------------------------------------ episodes (benchmark mode)
    def run_episodes(self):
        out = []
        for ep in range(self.args.episodes):
            rng = random.Random(self.args.seed + ep)
            if self.args.reset:
                print("reset:", self.bot.reset(rng.randrange(-2000, 2000), rng.randrange(-2000, 2000)), flush=True)
            s, reached, failed, steps, t0 = self.bot.state(), set(), set(), [], time.time()
            self.set_goal(self.args.goal)
            for t in range(self.args.max_steps):
                sk, trace, ms = self.goal_step(s)
                if sk is None or s["inventory"].get(self.args.goal, 0) > 0:
                    break
                print(f"      [plano {t:2d}] {skill_text(sk):50s} ({ms / 1000:.1f}s)", flush=True)
                self.update_dash(s, action=skill_text(sk), trace=trace)
                r = self.bot.act(sk)
                s2 = r["state"]
                stuck = (not r["ok"]) and s2["inventory"] == s["inventory"] and s2["near"] == s["near"]
                failed = (failed | {sk}) if stuck else set()
                print(f"                 -> {'ok  ' if r['ok'] else 'FAIL'} {r['msg']}  inv={s2['inventory']}", flush=True)
                steps.append({"t": t, "state": s, "skill": list(sk), "ok": r["ok"], "msg": r["msg"], "trace": trace, "decision_ms": ms})
                s = s2
                reached |= {i for i in MILESTONES if s["inventory"].get(i, 0) > 0}
            judged = [j for st in steps for j in st["trace"] if "truth" in j]
            e = {"episode": ep, "success": s["inventory"].get(self.args.goal, 0) > 0, "steps": steps,
                 "milestones": [m for m in MILESTONES if m in reached], "final_inventory": s["inventory"],
                 "seconds": time.time() - t0,
                 "predicate_acc": (sum(j["judged"] == j["truth"] for j in judged) / len(judged)) if judged else None,
                 "n_predicates": len(judged)}
            out.append(e)
            print(f"   success {e['success']}  milestones {len(e['milestones'])}/{len(MILESTONES)}  steps {len(steps)}  "
                  f"{e['seconds']:.0f}s" + (f"  predicate acc {e['predicate_acc']:.3f}" if e["predicate_acc"] else ""), flush=True)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="llama", choices=["llama", "torch"],
                    help="llama = llama.cpp server (fast here); torch = HF checkpoint in PyTorch (2.4 s/token on this CPU)")
    ap.add_argument("--server-url", default="http://127.0.0.1:8099", help="llama-server started with --pooling none")
    ap.add_argument("--api-key", default="", help="bearer token if the jev is served by somebody else (llama-server --api-key)")
    ap.add_argument("--head", default="models/jev-head.npz", help="3-class NLI head for the llama backend")
    ap.add_argument("--ckpt", default="models/openjev/qwen3.5-4b-nli-v2")
    ap.add_argument("--img-processor", default="models/qwen35-4b-proc")
    ap.add_argument("--goal", default="iron_pickaxe", choices=list(TREE))
    ap.add_argument("--after", default="", help="goal to switch to once --goal is done")
    ap.add_argument("--episodes", type=int, default=0, help=">0: run N scaffold episodes and exit (no chat)")
    ap.add_argument("--chat-only", action="store_true", help="never pursue the goal on its own: only obey orders")
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--phrasings", type=int, default=1)
    ap.add_argument("--rule", default="pair", choices=["pair", "threshold"])
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--image-size", type=int, nargs=2, default=[320, 240])
    ap.add_argument("--full-intents", action="store_true",
                    help="score all 19 intents per message (~75 s on this CPU) instead of the keyword-shortlisted few")
    ap.add_argument("--min-conf", type=float, default=0.55, help="below this the chat falls back to keywords")
    ap.add_argument("--min-margin", type=float, default=0.05)
    ap.add_argument("--skill-url", default="http://127.0.0.1:3010")
    ap.add_argument("--shot-url", default="http://127.0.0.1:3008/shot")
    ap.add_argument("--shots", default="shots")
    ap.add_argument("--dashboard", type=int, default=8080)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="start muted (chat replies still work)")
    ap.add_argument("--ignore", default="", help="comma-separated players whose chat is ignored (other bots)")
    ap.add_argument("--ignore-prefix", default="got it:", help="chat lines starting with this are treated as echoes")
    ap.add_argument("--out", default="logs/run.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(args.shots, exist_ok=True)
    agent = Agent(args)
    agent.boot()
    try:
        if args.episodes > 0:
            episodes = agent.run_episodes()
        else:
            agent.set_goal(args.goal)
            agent.run_forever()
            episodes = []
    except KeyboardInterrupt:
        episodes = []
    print("jev:", agent.jev.report(), flush=True)
    json.dump({"args": vars(args), "episodes": episodes, "chat": agent.chat.log,
               "model": {"calls": agent.jev.calls, "secs": agent.jev.secs, "load_s": agent.jev.load_s,
                         "ms_per_call": 1000 * agent.jev.secs / max(1, agent.jev.calls)}},
              open(args.out, "w"), indent=1)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
