#!/usr/bin/env python
"""Tiny stand-in for a llama-server: answers /props and /embedding but only with the right bearer token. Used to check
the client side of a remotely served jev (the real server cannot be duplicated here: two copies of the model do not fit
in this machine's RAM, the second one gets OOM-killed)."""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

KEY = sys.argv[2] if len(sys.argv) > 2 else "s3cret"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8097
HIDDEN = 2560


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _auth(self):
        if self.headers.get("authorization") != f"Bearer {KEY}":
            self.send_response(401)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"invalid api key","code":401}}')
            return False
        return True

    def do_GET(self):
        if not self._auth():
            return
        body = json.dumps({"media_marker": "<__media_test__>", "modalities": {"vision": True}}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if not self._auth():
            return
        n = len(self.rfile.read(int(self.headers.get("content-length", 0))))
        # one token sequence per requested prompt, deterministic vector so the client can be checked
        prompts = 1
        try:
            prompts = max(1, len(json.loads(n and b"1" or b"1"))) if False else 1
        except Exception:
            pass
        vec = [0.0] * HIDDEN
        vec[1] = 9.0   # makes the entailment class win: score head row 1
        body = json.dumps([{"index": i, "embedding": [vec]} for i in range(prompts)]).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print(f"fake jev on {PORT}, key {KEY}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
