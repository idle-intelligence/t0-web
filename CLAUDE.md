# t0-web — working rules

Distilled 2026-09-17 from the maintainer's global rules and the idle-intelligence repos (llm-web, stt-web, tts-web, sts-web, trucs.ai). This file is how we work.

## Code
- Rust. One crate tree, one source, two builds: native (`cli`/`native` feature) and `wasm32-unknown-unknown` (`web` feature). WASM is just another build target, never a port.
- Burn on wgpu via llm-web (Metal native, WebGPU in the tab). llm-web is an rlib dependency from a dedicated worktree + branch (`llm-life`); its Cargo manifests are frozen, its `mcp-agent` checkout is never touched.
- Backend/features (`metal`, `cuda`, `wgpu`, `accelerate`) are selected in the binary crate, not the library.
- No over-engineering. Only what the task needs. Three similar lines beat a premature abstraction. No error handling for things that can't happen.
- Don't add docstrings, comments or type annotations to code you didn't change. No premature optimization: get it working first.
- Pin pre-1.0 dependency versions (Burn, cubecl).
- Code is ground truth over docs. If a doc is stale, say so, don't trust it.
- `cargo clippy` clean before committing.

## Tests
- Tests exist for a reason: each one guards a behavior that was wrong or could plausibly break. No tests for the sake of coverage.
- `cargo test` must exit 0. No silent skips: a test that needs a file or a GPU says so and fails loudly when it's missing, or is explicitly ignored with a reason.
- Never claim something works without having run it. If it couldn't be run, say that.

## Git
- Commit early and often. Small, focused, atomic: one logical change per commit. Descriptive messages.
- Never set or change git identity. No Co-Authored-By on trivial commits.
- Never push, never open PRs, never publish models or datasets, unless the maintainer says so for that specific thing.
- `refs/` directories are read-only reference material.

## Demo
- The repo's demo page (`web/`) covers the repo's whole scope. It is self-contained: copy it to a static server and it works.
- Template: STATUS / INPUT / OUTPUT / PERFORMANCE panels. No settings panels, no URL inputs. Swap the engine, keep the template.
- trucs.ai is a separate thing with its own UX and choices. Not this repo's concern.
- Verify in Playwright's bundled headless Chromium. Never open or drive a personal browser. Drive the page through a small `window.__app` API, not DOM clicks.

## Models, data, HF
- Weights and datasets are never committed. Fetch at runtime or regenerate.
- Local models live under `~/Code/idle-intelligence/models/`, never `~/models`.
- The HF CLI is `hf` (never `huggingface-cli`). Upload: `hf upload <repo_id> <local_path> <path_in_repo>` — only when told.

## GPU and measurements
- One GPU job at a time on the local machine unless the maintainer says otherwise for the night. Never report throughput measured under contention; mark such numbers provisional.
- Benchmarks and training runs are a research log: machine, commit, exact command, then a data table. Tables are data only; analysis goes in a separate doc. One results file per run, not appended.
- Cite sources (arxiv, GitHub) in research docs.

## Agents
- Lead orchestrates, workers do one task each, scoped to one repo. Worker reports ≤ 600 words. No nested spawning.
- Multi-repo: never edit another repo from this one.
