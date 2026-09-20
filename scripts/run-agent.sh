#!/bin/sh
# The NPC: chat-driven jev agent for the survival bot.
#   scripts/run-agent.sh                  # obeys chat orders, keeps the goal when idle
#   scripts/run-agent.sh --chat-only      # does nothing on its own, only obeys
set -e
cd "$(dirname "$0")/.."

exec ./venv/bin/python -u -m jev.agent \
  --backend llama \
  --server-url "${SERVER_URL:-http://127.0.0.1:8099}" \
  ${API_KEY:+--api-key "$API_KEY"} \
  --vision \
  --dashboard "${DASHBOARD:-8080}" \
  --threads "${THREADS:-10}" \
  "$@"
