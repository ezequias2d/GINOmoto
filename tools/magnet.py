#!/usr/bin/env python
"""Print the infohash and the magnet link of the distribution torrent."""
from __future__ import annotations

import os
from urllib.parse import quote

from torrentool.api import Torrent

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
t = Torrent.from_file(os.path.join(BASE, "dist/ginomoto-jev-q4-km.torrent"))
TRACKERS = ["udp://tracker.opentrackr.org:1337/announce", "udp://open.demonii.com:1337/announce",
            "udp://tracker.openbittorrent.com:6969/announce", "udp://exodus.desync.com:6969/announce"]
magnet = f"magnet:?xt=urn:btih:{t.info_hash}&dn={t.name}&tr=" + "&tr=".join(quote(x, safe="") for x in TRACKERS)
with open(os.path.join(BASE, "dist/MAGNET.txt"), "w") as fh:
    fh.write(magnet + "\n")
print("infohash:", t.info_hash)
print("name:", t.name)
print("magnet:", magnet)
