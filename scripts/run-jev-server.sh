#!/bin/sh
# jev's backbone: llama.cpp serving the openjev GGUF with pooling=none, so /embedding returns the per-token hidden
# states and the NLI head (models/jev-head.npz) is applied by jev/model_llama.py.
#
#   scripts/run-jev-server.sh            # iGPU (Vulkan) for the matmuls
#   NGL=0 scripts/run-jev-server.sh      # CPU only
set -e
cd "$(dirname "$0")/.."

LLAMA_DIR="${LLAMA_CPP:-$HOME/llama.cpp}"
BIN="$LLAMA_DIR/build/bin/llama-server"
MODEL="${MODEL:-models/jev/jev-qwen35-4b-f16.gguf}"
MMPROJ="${MMPROJ:-models/jev/mmproj-jev-qwen35-4b-f16.gguf}"

[ -x "$BIN" ] || { echo "no llama-server at $BIN (build it: scripts/build-llama.sh)"; exit 1; }

# -ub has to fit an image prompt: one 640x360 frame is ~900 vision tokens in Qwen3.5's tower, and llama-server aborts
# any prompt bigger than the physical batch (`--image-max-tokens` caps how many the frame is allowed to become).
ARGS="-m $MODEL --host 127.0.0.1 --port ${PORT:-8099} -c ${CTX:-8192} -np ${PARALLEL:-8} -ngl ${NGL:-99} -t ${THREADS:-10} \
      -b ${BATCH:-2048} -ub ${UBATCH:-2048} --image-max-tokens ${IMAGE_MAX_TOKENS:-512} \
      --embeddings --pooling none ${EXTRA_ARGS:-}"
[ -f "$MMPROJ" ] && ARGS="$ARGS --mmproj $MMPROJ"
echo "llama-server $ARGS"
exec "$BIN" $ARGS
