#!/bin/sh
# Build llama.cpp (CPU + Vulkan) the converter and server come from. Vulkan SDK path is this machine's.
set -e
LLAMA_DIR="${LLAMA_CPP:-$HOME/llama.cpp}"
[ -d "$LLAMA_DIR" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
cd "$LLAMA_DIR"
export VULKAN_SDK="${VULKAN_SDK:-/opt/vulkan-sdk/x86_64}"
export PATH="$VULKAN_SDK/bin:$PATH"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF
cmake --build build -j "${JOBS:-8}" --target llama-server llama-cli llama-mtmd-cli
echo "built: $LLAMA_DIR/build/bin/llama-server"
