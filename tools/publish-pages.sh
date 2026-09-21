#!/usr/bin/env bash
# Build both wasm engines and publish the committed HEAD's web/ demo to an
# orphan `gh-pages` branch, following the layout used by ../tts-web and
# ../stt-web (repo root = the served tree, no crates/ or models/).
#
# web/worker.js resolves the wasm module at `${PKG_DIR}/t0_wasm.js`
# relative to itself, where PKG_DIR is picked at runtime:
#   ./pkg-wgpu  when a WebGPU adapter is available  -> the `fast` build
#               (crates/t0-fast, WGSL kernels, the engine the benchmarks
#               and the model cards quote)
#   ./pkg       otherwise                            -> the `ndarray` build
#               (Burn CPU backend)
# Both are published so every browser gets a working path.
#
# Also publishes bench/compare/index.html and bench/compare/compare.js
# (only those two files -- that page loads its models from Hugging Face
# and its onnxruntime-web build from jsdelivr, never from gitignored
# bench/onnx or bench/ours).
#
# Never checks out gh-pages in the main working tree; never pushes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WORKTREE_DIR="$(mktemp -d)/gh-pages-worktree"
EXPORT_DIR="$(mktemp -d)/web-export"

cleanup() {
    git worktree remove --force "$WORKTREE_DIR" >/dev/null 2>&1 || true
    rm -rf "$EXPORT_DIR"
}
trap cleanup EXIT

echo "==> Building t0-wasm (fast feature, WebGPU engine)"
wasm-pack build crates/t0-wasm --target web --out-dir pkg-fast --release --no-default-features --features fast
echo "==> Building t0-wasm (ndarray feature, CPU engine)"
wasm-pack build crates/t0-wasm --target web --out-dir pkg --release

FAST_SRC="$REPO_ROOT/crates/t0-wasm/pkg-fast"
CPU_SRC="$REPO_ROOT/crates/t0-wasm/pkg"
for d in "$FAST_SRC" "$CPU_SRC"; do
    if [ ! -f "$d/t0_wasm.js" ] || [ ! -f "$d/t0_wasm_bg.wasm" ]; then
        echo "error: expected build output not found in $d" >&2
        exit 1
    fi
done

echo "==> Exporting committed HEAD's web/ and bench/compare/ into $EXPORT_DIR"
mkdir -p "$EXPORT_DIR"
git archive HEAD web | tar -x -C "$EXPORT_DIR"
mkdir -p "$EXPORT_DIR/bench/compare"
git archive HEAD -- bench/compare/index.html bench/compare/compare.js | tar -x -C "$EXPORT_DIR"

place() { # src dir, dest dir
    rm -rf "$2"; mkdir -p "$2"
    cp "$1/t0_wasm.js" "$1/t0_wasm_bg.wasm" "$2/"
    [ -f "$1/package.json" ] && cp "$1/package.json" "$2/"
}
echo "==> Placing the fast build at web/pkg-wgpu and the ndarray build at web/pkg"
place "$FAST_SRC" "$EXPORT_DIR/web/pkg-wgpu"
place "$CPU_SRC" "$EXPORT_DIR/web/pkg"

# web/models is gitignored (weights are fetched from Hugging Face and
# cached by the browser, never shipped); git archive already excludes it.
if [ -d "$EXPORT_DIR/web/models" ]; then
    echo "error: unexpected web/models in export -- refusing to publish weights" >&2
    exit 1
fi

echo "==> Preparing orphan gh-pages worktree at $WORKTREE_DIR"
mkdir -p "$(dirname "$WORKTREE_DIR")"
if git show-ref --verify --quiet refs/heads/gh-pages; then
    git worktree add "$WORKTREE_DIR" gh-pages
else
    git worktree add --detach "$WORKTREE_DIR" HEAD
    git -C "$WORKTREE_DIR" checkout --orphan gh-pages
    git -C "$WORKTREE_DIR" rm -rf . >/dev/null 2>&1 || true
fi

find "$WORKTREE_DIR" -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +
cp -R "$EXPORT_DIR/web" "$WORKTREE_DIR/web"
mkdir -p "$WORKTREE_DIR/bench/compare"
cp "$EXPORT_DIR/bench/compare/index.html" "$EXPORT_DIR/bench/compare/compare.js" "$WORKTREE_DIR/bench/compare/"

cd "$WORKTREE_DIR"
git add -A
if git diff --cached --quiet; then
    echo "==> No changes; gh-pages already up to date"
else
    git commit -q -m "Publish web/ demo (WebGPU and CPU engines) to GitHub Pages"
fi
cd "$REPO_ROOT"

echo "==> gh-pages tip: $(git rev-parse gh-pages)"
