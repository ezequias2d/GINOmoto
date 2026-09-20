#!/usr/bin/env python
"""Seed the GINOmoto checkpoint torrent from this machine.

    python3.13 tools/seed.py            # venv com libtorrent (uv venv --python 3.13 && uv pip install libtorrent)

Files must already be in place (seed_mode: the payload is 'done' and only uploaded). Prints peers/uploaded so it is
visible whether anybody is actually pulling — if the network is behind NAT without UPnP, P2P may stay silent and the
HTTP link next to the magnet is the one that works.
"""
from __future__ import annotations

import os
import sys
import time

import libtorrent as lt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TORRENT = os.path.join(BASE, "dist/ginomoto-jev-q4-km.torrent")
SAVE = os.path.join(BASE, "dist")
PORT = int(os.environ.get("P2P_PORT", 6881))


def main():
    settings = {
        "listen_interfaces": f"0.0.0.0:{PORT},[::]:{PORT}",
        "enable_dht": True,
        "enable_lsd": True,
        "enable_upnp": True,
        "enable_natpmp": True,
        "announce_to_all_trackers": True,
        "announce_to_all_tiers": True,
        "alert_mask": lt.alert.category_t.error_notification,
    }
    sess = lt.session(settings)
    sess.add_dht_router("router.bittorrent.com", 6881)
    sess.add_dht_router("dht.transmissionbt.com", 6881)
    sess.add_dht_router("router.utorrent.com", 6881)
    sess.start_dht()

    atp = lt.add_torrent_params()
    atp.ti = lt.torrent_info(TORRENT)
    atp.save_path = SAVE
    atp.flags |= lt.torrent_flags.seed_mode | lt.torrent_flags.auto_managed
    h = sess.add_torrent(atp)
    h.force_recheck()
    print(f"seeding {h.torrent_file().name()} | infohash {h.info_hash()}", flush=True)
    print(f"listening on {PORT} (udp+tcp), upnp={settings['enable_upnp']} natpmp={settings['enable_natpmp']}", flush=True)
    print("host is IPv6-first (IPv4 behind CGNAT): direct peers work over IPv6, otherwise the HTTP link is the way",
          flush=True)

    while True:
        time.sleep(30)
        st = h.status()
        ss = sess.status()
        print(f"  {lt.torrent_status.states(st.state)} | peers {st.num_peers} seeds {st.num_seeds} | "
              f"uploaded {st.total_upload / 1e9:.2f} GB | rate {st.upload_rate / 1024:.0f} KiB/s | "
              f"dht nodes {ss.dht_nodes} | listen ok {ss.has_incoming_connections}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
