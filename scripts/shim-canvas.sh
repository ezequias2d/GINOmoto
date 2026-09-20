#!/bin/sh
# prismarine-viewer hard-requires the native 'canvas' package, which needs cairo+pango dev headers.
# This machine has cairo but no pango, so point 'canvas' at @napi-rs/canvas (prebuilt napi binary, same API).
set -e
cd "$(dirname "$0")/.."
npm i @napi-rs/canvas --no-audit --no-fund
mkdir -p node_modules/canvas
echo '{"name":"canvas","version":"0.0.0","main":"index.js"}' > node_modules/canvas/package.json
echo "module.exports = require('@napi-rs/canvas')" > node_modules/canvas/index.js
node -e "require('canvas').createCanvas(4,4); console.log('canvas shim ok')"
