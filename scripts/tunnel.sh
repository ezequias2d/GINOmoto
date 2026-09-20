#!/bin/sh
# Public TCP tunnel so friends can join the local Paper server. bore.pub assigns a random port on every (re)connect: the
# address is whatever the journal says ("listening at bore.pub:PORT"). No account, no credit card.
BORE="$(command -v bore || echo "$HOME/.local/bin/bore")"
exec "$BORE" local "${PORT:-25565}" --to bore.pub
