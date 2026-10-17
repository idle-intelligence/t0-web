# Multithreaded CPU (ndarray) build: feasibility spike (stopped after phase 1)

Machine: Apple M2 (Darwin 25.3.0, 4 performance + 4 efficiency cores,
`navigator.hardwareConcurrency` = 8), Playwright's bundled
Chromium-for-Testing `chromium-1243` (Chrome for Testing 153.0.8010.12),
headless.

## How burn-ndarray parallelises

`burn-ndarray` 0.20.1 has a `multi-threads` feature (`rayon` +
`ndarray/rayon` + `matrixmultiply/threading`), off by default and off in
this workspace (`burn-ndarray = { workspace = true, optional = true }`,
`default-features = false` implied by the workspace dependency, no
`multi-threads`). Enabling it does reach `rayon` even on
`wasm32-unknown-unknown` -- `rayon` 1.12.0 was already resolvable and
built clean for wasm once the standard `wasm-bindgen-rayon` thread-pool
shim supplied the `std::thread::spawn` implementation over Web Workers.
So the "no rayon in wasm" stop condition did not apply here; the spike
went on to build and measure.

## Toolchain

- Installed `rustup toolchain install nightly --component rust-src`
  (`rustc 1.100.0-nightly (bba531001 2026-09-20)`); none was present
  before.
- `wasm-bindgen-rayon` 1.3.0 depends on `wasm-bindgen ^0.2.99`; the
  workspace lock has `wasm-bindgen 0.2.128`, compatible.
- The installed `wasm-bindgen-cli` was 0.2.108, mismatched with the
  lockfile's 0.2.128 (wasm-bindgen refuses to run against a
  mismatched-version wasm binary); reinstalled with `cargo install
  wasm-bindgen-cli --version 0.2.128 --locked`.
- `wasm-pack` cannot be used for this build: it appends extra args after
  `--` to the `cargo build` invocation, which places `-Z
  build-std=panic_abort,std` after cargo's own `--`, where cargo treats
  it as a rustc argument, not a cargo flag, and errors
  (`unexpected argument '-Z' found`). `tools/build-mt.sh` drives `cargo
  build` (nightly) + `wasm-bindgen` (CLI) directly instead.

## Build recipe that worked

Added a `threads` feature to `crates/t0-wasm` (`burn-ndarray/multi-threads`
+ `dep:wasm-bindgen-rayon`, re-exporting `wasm_bindgen_rayon::init_thread_pool`
from `lib.rs`), built via `tools/build-mt.sh` into
`crates/t0-wasm/pkg-mt`. Final `RUSTFLAGS` needed, arrived at by working
through three successive linker errors:

```
-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128
-C link-arg=--shared-memory -C link-arg=--max-memory=1073741824
-C link-arg=--import-memory
-C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size
-C link-arg=--export=__tls_align -C link-arg=--export=__tls_base
```

plus `cargo +nightly build ... -Z build-std=panic_abort,std`. Notes:

- `+simd128` has to be repeated here: `.cargo/config.toml`'s `rustflags`
  (which is where the single-thread `pkg` build gets `+simd128` from) is
  replaced, not merged, by an explicit `RUSTFLAGS` env var, so the first
  threaded build silently lost SIMD and was not a fair comparison against
  `pkg` until this was added back.
- `--shared-memory`/`--max-memory`/`--import-memory` are needed for LLD to
  emit an actual `shared` wasm `Memory` import; without them
  `wasm-bindgen`'s thread-splitting pass errors first with "threading
  requires memory to be imported", then (once `--import-memory` is added)
  the built module still can't run: browsers refuse to `postMessage` a
  non-shared `WebAssembly.Memory` to a worker (`DataCloneError`).
- The `--export=__wasm_init_tls`/`__tls_*` flags are needed or
  `wasm-bindgen`'s threading pass fails with "failed to find
  `__wasm_init_tls`" -- LLD strips these symbols in a release build unless
  told to keep them.
- `wasm-bindgen-rayon`'s generated `snippets/.../workerHelpers.js` does
  `await import('../../..')` to reach the main `pkg` module, which is a
  bundler-relative resolution (consulting `package.json`'s `main`) and
  does not work for a plain browser `import()` of a bare directory
  (`--target web`, no bundler, matching how `web/worker.js` already loads
  `pkg`/`pkg-fast`) -- fails with "Failed to fetch dynamically imported
  module". Patched to `await import('../../../t0_wasm.js')` in
  `tools/build-mt.sh`'s output after `wasm-bindgen` runs; not upstreamed,
  just a local post-process since we are not wiring this build into
  `web/` this round.

## Benchmark

Harness: `/tmp/t0-cpu-threads-bench/bench.html` (adapted from
`~/.claude/jobs/f13f376f/tmp/bench.html`, adding an `initThreadPool(n)`
call before `initBackend()` when a `threads` query param is set) +
`/tmp/t0-cpu-threads-bench/run_bench.mjs` (same Playwright driver, extra
`threads`/`port` args), served from the repo root by
`/tmp/t0-cpu-threads-bench/coi_server.py` (`http.server` subclass adding
`Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` on every response --
required for `SharedArrayBuffer`/atomics and for `self.crossOriginIsolated`
to read `true`). Alpha Q4_0, context 512 points
(`web/data/series.f32[3088:3600]`), horizon 32, 3 warm-ups, median and p90
of 10 timed `forecast()` calls.

| build | threads | median (ms) | p90 (ms) |
|---|---|---|---|
| `web/pkg` (single-thread, SIMD) | - | 235.1 | 242.3 |
| `web/pkg-mt` (SIMD + atomics, `threads` feature) | 8 (hardwareConcurrency) | 244.1 | 247.0 |
| `web/pkg-mt` (SIMD + atomics, `threads` feature) | 4 (phone-like) | 252.2 | 255.1 |

Agreement: max abs difference across all 160 output values (horizon 32 x
n_quantiles 5) was 0.0009765625 for both thread counts against the
single-thread build -- ordinary float-reordering noise, not a
correctness issue.

## Decision

Both thread counts are *slower* than the single-thread SIMD build (0.93x
and 0.96x of baseline, i.e. a slowdown, not a speedup) -- nowhere near the
1.3x-at-4-threads bar for proceeding to phase 2. `initThreadPool()` itself
adds ~0 to `loadMs` here (pool spin-up happens before the timed loop), so
the loss shows up entirely in `forecast()`: the rayon/web-worker dispatch
overhead per call outweighs any parallel work at this model's matmul
sizes (context 512, single Q4_0 alpha checkpoint, horizon-32 batch of 1).
Stopping here per the phase 1 gate -- no `web/worker.js`, no
`tools/publish-pages.sh`, no trucs.ai `_headers` changes in this round.

`crates/t0-wasm`'s `threads` Cargo feature and `tools/build-mt.sh` are
kept (feature-gated, opt-in, no effect on the default `ndarray`/`fast`
builds -- verified with `cargo check -p t0-wasm --target
wasm32-unknown-unknown`, unaffected) as a working recipe, in case a future
model size or workload shape (larger batch, longer context, a bigger
checkpoint) changes this calculus enough to be worth revisiting.
