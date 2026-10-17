#!/usr/bin/env bash
# Build the multithreaded CPU (ndarray) wasm engine into
# crates/t0-wasm/pkg-mt. Separate from the default single-thread build:
# the atomics/bulk-memory/mutable-globals target features and the
# `-Z build-std` nightly flag only apply to this invocation (never to
# .cargo/config.toml, which would break the single-thread pkg build).
#
# Requires:
#   - a nightly toolchain with rust-src: `rustup toolchain install nightly
#     --component rust-src`
#   - a wasm-bindgen-cli matching the wasm-bindgen version in Cargo.lock
#     exactly (`cargo install wasm-bindgen-cli --version <that version>
#     --locked`) -- wasm-pack cannot pass `-Z build-std` through to cargo
#     (it lands after `--`, where cargo treats it as a rustc flag), so
#     this script drives `cargo build` + `wasm-bindgen` directly instead
#     of going through wasm-pack.
#
# Output is the ndarray backend built with `burn-ndarray/multi-threads`
# (rayon) and `wasm-bindgen-rayon` for the worker-pool thread spawn shim.
# The page must call the exported `initThreadPool(n)` before `initBackend`
# and must be served cross-origin isolated (COOP/COEP) for
# SharedArrayBuffer + atomics to work at all.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="crates/t0-wasm/pkg-mt"

echo "==> Building t0-wasm (threads feature, nightly build-std)"
RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals \
-C link-arg=--shared-memory -C link-arg=--max-memory=1073741824 \
-C link-arg=--import-memory \
-C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size \
-C link-arg=--export=__tls_align -C link-arg=--export=__tls_base" \
  cargo +nightly build -p t0-wasm \
    --target wasm32-unknown-unknown \
    --release \
    --no-default-features --features threads \
    -Z build-std=panic_abort,std

WASM_IN="target/wasm32-unknown-unknown/release/t0_wasm.wasm"
if [ ! -f "$WASM_IN" ]; then
    echo "error: expected build output not found at $WASM_IN" >&2
    exit 1
fi

echo "==> Running wasm-bindgen"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
wasm-bindgen --target web --out-dir "$OUT_DIR" --out-name t0_wasm "$WASM_IN"

echo "==> Wrote $OUT_DIR"
