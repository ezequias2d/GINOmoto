"""Client for the mineflayer skill server (bot/skill_server.js) — same state/action format as the repo's MineflayerEnv."""
from __future__ import annotations

import json
import urllib.request


class SkillBot:
    def __init__(self, url: str = "http://127.0.0.1:3010"):
        self.url, self.since = url, 0

    def _call(self, path, payload=None, timeout=300):
        req = urllib.request.Request(self.url + path, data=None if payload is None else json.dumps(payload).encode(),
                                     headers={"content-type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

    def state(self):
        return self._call("/state", timeout=30)

    def chat(self, since: int | None = None):
        """New player messages since `since` (default: since the last poll). Pass `since=0` to start clean."""
        if since is None:
            since = self.since
        r = self._call(f"/chat?since={int(since)}", timeout=30)
        self.since = r.get("now", self.since)
        return r.get("messages", [])

    def act(self, sk):
        kind, arg, n = sk
        r = self._call("/act", {"skill": kind, "arg": arg, "n": n})
        return {"ok": r["ok"], "msg": r["msg"],
                "state": {k: r["state"][k] for k in ("inventory", "near", "visible", "pos", "players") if k in r["state"]}}

    def reset(self, x: int, z: int):
        return self._call("/reset", {"x": x, "z": z}, timeout=180)["msg"]

    def say(self, text: str):
        return self._call("/act", {"skill": "say", "arg": text, "n": 1})

    def wait_ready(self, tries: int = 40, delay: float = 1.0):
        import time
        for _ in range(tries):
            try:
                s = self.state()
                if s:
                    return s
            except Exception:
                pass
            time.sleep(delay)
        raise RuntimeError(f"skill server at {self.url} never became ready")


def shot(out: str, url: str = "http://127.0.0.1:3008/shot") -> str:
    """Ask the screenshot service for a fresh frame of the bot's first-person view."""
    req = urllib.request.Request(url, data=json.dumps({"out": out}).encode(), headers={"content-type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=90).read())
    if not r.get("ok"):
        raise RuntimeError(f"screenshot failed: {r}")
    return r["file"]
