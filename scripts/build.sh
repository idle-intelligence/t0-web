#!/usr/bin/env bash
# Builds both t0-wasm engines (fast/WebGPU and ndarray/CPU) and assembles the
# deployed site into _site/: web/ (the demo) plus bench/compare/ (the
# compare page, which loads its own models from Hugging Face and onnxruntime
# from jsdelivr, never from bench/onnx or bench/ours).
#
# Rewrites the ENGINE_BUILD tag on every wasm loading URL (web/worker.js,
# bench/compare/compare.js) and checks neither built wasm carries a local
# build path.
#
# Requires wasm-pack and wasm-bindgen-cli (version matching Cargo.lock's
# wasm-bindgen exactly: `cargo install wasm-bindgen-cli --version <ver>
# --locked`).
#
# Usage: ENGINE_BUILD=<tag> scripts/build.sh
# ENGINE_BUILD defaults to "dev" for local builds; CI passes the commit sha.
set -euo pipefail

ENGINE_BUILD="${ENGINE_BUILD:-dev}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Building t0-wasm (fast feature, WebGPU engine) for ENGINE_BUILD=$ENGINE_BUILD"
RUSTFLAGS="--remap-path-prefix=$HOME=/home" \
  wasm-pack build crates/t0-wasm --target web --out-dir pkg-fast --release --no-default-features --features fast

echo "==> Building t0-wasm (ndarray feature, CPU engine)"
RUSTFLAGS="--remap-path-prefix=$HOME=/home" \
  wasm-pack build crates/t0-wasm --target web --out-dir pkg --release

FAST_WASM="crates/t0-wasm/pkg-fast/t0_wasm_bg.wasm"
CPU_WASM="crates/t0-wasm/pkg/t0_wasm_bg.wasm"

# --- Local-path / user-name leak check on built wasm outputs ---
for WASM in "$FAST_WASM" "$CPU_WASM"; do
    LEAKS=$(strings "$WASM" | grep -F -e "$HOME" -e "Code/" -e ".claude/" -e "/Users/" || true)
    USER_HITS=$(strings "$WASM" | grep -Fw -e "$(id -un)" || true)
    if [ -n "$LEAKS$USER_HITS" ]; then
        echo "error: $WASM contains local paths or the user name:" >&2
        printf '%s\n%s\n' "$LEAKS" "$USER_HITS" | grep -v '^$' | head -20 >&2
        exit 1
    fi
done
echo "==> no local paths in built wasm"

# --- Assemble the deployed site into _site/ ---
echo "==> Assembling _site"
rm -rf _site
mkdir -p _site/bench/compare
cp -R web _site/web
cp bench/compare/index.html bench/compare/compare.js _site/bench/compare/

place() { # src dir, dest dir
    rm -rf "$2"; mkdir -p "$2"
    cp "$1/t0_wasm.js" "$1/t0_wasm_bg.wasm" "$2/"
    [ -f "$1/package.json" ] && cp "$1/package.json" "$2/"
}
rm -rf _site/web/pkg-wgpu _site/web/pkg _site/web/pkg-mt _site/web/models
echo "==> Placing the fast build at web/pkg-wgpu and the ndarray build at web/pkg"
place "$REPO_ROOT/crates/t0-wasm/pkg-fast" "_site/web/pkg-wgpu"
place "$REPO_ROOT/crates/t0-wasm/pkg" "_site/web/pkg"

if [ -d _site/web/models ]; then
    echo "error: unexpected web/models in export -- refusing to publish weights" >&2
    exit 1
fi

# --- Rewrite the ENGINE_BUILD tag on every loading URL ---
echo "==> Rewriting ENGINE_BUILD tag to $ENGINE_BUILD"
sed -i.bak "s/const ENGINE_BUILD = '[^']*'/const ENGINE_BUILD = '$ENGINE_BUILD'/" \
  _site/web/worker.js _site/bench/compare/compare.js
rm -f _site/web/worker.js.bak _site/bench/compare/compare.js.bak

COUNT="$(grep -rEo "ENGINE_BUILD = '[^']*'" _site/web/worker.js _site/bench/compare/compare.js | grep -Fc "'$ENGINE_BUILD'")"
if [ "$COUNT" -ne 2 ]; then
    echo "error: expected 2 ENGINE_BUILD assignments rewritten to $ENGINE_BUILD, found $COUNT" >&2
    exit 1
fi

echo "==> Wrote _site"
