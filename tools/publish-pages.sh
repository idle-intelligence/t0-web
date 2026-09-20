#!/usr/bin/env bash
# Build the fast/CPU wasm backend and publish the committed HEAD's web/
# demo to an orphan `gh-pages` branch, following the layout used by
# ../stt-web (repo root = the served tree, no crates/ or models/,
# no .nojekyll -- stt-web's own gh-pages doesn't carry one either).
#
# t0-web's web/worker.js resolves the wasm module at `${PKG_DIR}/t0_wasm.js`
# relative to itself (PKG_DIR is './pkg' or './pkg-wgpu', picked at runtime
# by navigator.gpu -- see crates/t0-wasm/README.md), unlike stt-web where
# pkg/ sits one level up from web/. This script only builds the `fast`
# feature (single CPU backend, no Burn), so it publishes that build as
# `web/pkg` -- the path worker.js already falls back to when
# navigator.gpu is absent. No edit to worker.js is needed. Browsers that
# do expose navigator.gpu will still request `web/pkg-wgpu`, which this
# script does not build or publish; that's a known gap, not a bug in this
# script (the task only asked for the `fast` build).
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

echo "==> Building t0-wasm (fast feature, release)"
wasm-pack build crates/t0-wasm --target web --out-dir pkg-fast --release --no-default-features --features fast

PKG_SRC="$REPO_ROOT/crates/t0-wasm/pkg-fast"
if [ ! -f "$PKG_SRC/t0_wasm.js" ] || [ ! -f "$PKG_SRC/t0_wasm_bg.wasm" ]; then
    echo "error: expected build output not found in $PKG_SRC" >&2
    exit 1
fi

echo "==> Exporting committed HEAD's web/ into $EXPORT_DIR"
mkdir -p "$EXPORT_DIR"
git archive HEAD web | tar -x -C "$EXPORT_DIR"

echo "==> Placing pkg-fast build at web/pkg (the path worker.js resolves when navigator.gpu is absent)"
rm -rf "$EXPORT_DIR/web/pkg"
mkdir -p "$EXPORT_DIR/web/pkg"
cp "$PKG_SRC/t0_wasm.js" "$PKG_SRC/t0_wasm_bg.wasm" "$EXPORT_DIR/web/pkg/"
[ -f "$PKG_SRC/package.json" ] && cp "$PKG_SRC/package.json" "$EXPORT_DIR/web/pkg/"

# web/models is gitignored (weights fetched/cached at runtime by the
# browser, not shipped) -- git archive of HEAD already excludes it, and
# web/data (the bundled series fixture) is small and kept, matching
# stt-web's web/test-bria.wav being committed as demo fixture data.
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

# Replace the worktree's tracked content with the fresh export (root of
# the branch = the served tree, mirroring stt-web's gh-pages layout).
find "$WORKTREE_DIR" -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +
cp -R "$EXPORT_DIR/web" "$WORKTREE_DIR/web"

cd "$WORKTREE_DIR"
git add -A
if git diff --cached --quiet; then
    echo "==> No changes; gh-pages already up to date"
else
    git commit -q -m "Publish web/ demo (t0-alpha, fast/CPU backend) to GitHub Pages"
fi
cd "$REPO_ROOT"

TIP="$(git rev-parse gh-pages)"
echo "==> gh-pages tip: $TIP"
