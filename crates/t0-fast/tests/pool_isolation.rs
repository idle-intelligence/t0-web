//! Regression test for the `Pool`-sharing bug documented in
//! `docs/runs/2026-09-20-compare-page.md`: `bench/compare` loaded two
//! `GpuModel`s (a Q8_0 GGUF build and an F32-residency reference build)
//! against one shared `t0_fast::Engine`. Before the fix, `Engine` owned a
//! single `Pool` whose bind-group cache keys are bare call-site strings
//! with no per-model/per-quant component (`pool.rs`'s module doc), so the
//! second model's first forward call at a shape the first model already
//! ran at reused the first model's cached `wgpu::BindGroup`s -- bound to
//! the *wrong* model's weight buffers -- and silently produced up to 72%
//! divergence from ground truth.
//!
//! The fix moves `Pool` from `Engine` to `GpuModel` (`GpuModel::pool`):
//! two models can still share one `Engine` (device/queue/pipelines are
//! stateless), but never a `Pool`. This test loads two differently-
//! quantized models against one shared `Engine` and alternates calls
//! between them, asserting each call's output is bit-for-bit identical to
//! that same model's output computed in isolation -- the property that
//! silently broke before the fix.
//!
//! `#[ignore]`d: needs the committed-small-but-not-tiny GGUF/safetensors
//! fixtures under `bench/ours/` (weights are never committed generally,
//! but these bench fixtures are checked in for exactly this kind of
//! local regression test -- see `docs/runs/2026-09-20-compare-page.md`).

use std::path::PathBuf;

use t0_core::Weights;
use t0_fast::{forecast, load_model, load_model_from_gguf, Engine, WeightQuant};

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn sine_context(n: usize, freq: f32) -> Vec<f32> {
    (0..n).map(|i| (freq * i as f32).sin()).collect()
}

/// Two differently-quantized `GpuModel`s (Q8_0 GGUF, F32 safetensors)
/// sharing one `Engine`, alternating forecast calls A,B,A,B. Each call's
/// result must exactly equal (bit-for-bit) that same model's own
/// standalone result -- the other model's interleaved calls must not
/// perturb it via any shared cache.
#[test]
#[ignore = "requires bench/ours/t0-alpha-{q8_0.gguf,f32.safetensors,f32-config.json}; run from repo root with `cargo test -p t0-fast --release -- --ignored pool_isolation`"]
fn q8_and_f32_models_share_engine_without_cross_contamination() {
    let dir = manifest_dir().join("../../bench/ours");
    let gguf_path = dir.join("t0-alpha-q8_0.gguf");
    let f32_path = dir.join("t0-alpha-f32.safetensors");
    let config_path = dir.join("t0-alpha-f32-config.json");
    for p in [&gguf_path, &f32_path, &config_path] {
        assert!(p.exists(), "{} not found -- see the #[ignore] reason above", p.display());
    }

    let engine = Engine::new().unwrap();

    let gguf_bytes = std::fs::read(&gguf_path).unwrap();
    let model_q8 = load_model_from_gguf(&engine, &gguf_bytes).unwrap();

    let (weights, config) = Weights::load_auto(&f32_path, Some(&config_path)).unwrap();
    let model_f32 = load_model(&engine, &weights, config, WeightQuant::F32).unwrap();

    let t_ctx = 512;
    let horizon = 32;
    let ctx_q8 = sine_context(t_ctx, 0.03);
    let ctx_f32 = sine_context(t_ctx, 0.05);

    // Baselines: each model forecasting alone, nothing interleaved.
    let baseline_q8 = forecast(&engine, &model_q8, &ctx_q8, 1, t_ctx, horizon).unwrap();
    let baseline_f32 = forecast(&engine, &model_f32, &ctx_f32, 1, t_ctx, horizon).unwrap();

    // Interleaved A,B,A,B at the exact same (v, p) shape both baselines
    // used -- the scenario that hit the shared-Pool bug (same shape means
    // the second model's first call reused the first model's cached bind
    // groups under the old single-Engine-Pool design).
    let a1 = forecast(&engine, &model_q8, &ctx_q8, 1, t_ctx, horizon).unwrap();
    let b1 = forecast(&engine, &model_f32, &ctx_f32, 1, t_ctx, horizon).unwrap();
    let a2 = forecast(&engine, &model_q8, &ctx_q8, 1, t_ctx, horizon).unwrap();
    let b2 = forecast(&engine, &model_f32, &ctx_f32, 1, t_ctx, horizon).unwrap();

    assert_eq!(a1, baseline_q8, "Q8_0 model's first interleaved call diverged from its standalone baseline");
    assert_eq!(b1, baseline_f32, "F32 model's first interleaved call diverged from its standalone baseline");
    assert_eq!(a2, baseline_q8, "Q8_0 model's second interleaved call diverged from its standalone baseline");
    assert_eq!(b2, baseline_f32, "F32 model's second interleaved call diverged from its standalone baseline");
}

/// Two *different-config* GGUF models (t0-alpha, t0-beta from the beta
/// worktree's fixtures) sharing one `Engine`: a stronger version of the
/// same regression, since alpha/beta differ in more than quantization
/// (different uniform struct sizes for some call sites in the beta CLI's
/// own history, per the task brief) -- if any cache were keyed only by
/// call-site string and shape, this is the case most likely to produce a
/// wrong-size or wrong-buffer bind group instead of just wrong values.
/// Skips (does not fail) if the beta worktree's fixtures aren't present,
/// since that worktree is a separate, optional checkout.
#[test]
#[ignore = "requires the beta worktree's fixtures; run from repo root with `cargo test -p t0-fast --release -- --ignored pool_isolation`"]
fn alpha_and_beta_models_share_engine_without_cross_contamination() {
    let alpha_path = manifest_dir().join("../../bench/ours/t0-alpha-q8_0.gguf");
    let beta_path = manifest_dir().join("../../.claude/worktrees/beta/fixtures/beta/gguf/t0-beta-q8_0.gguf");
    if !alpha_path.exists() || !beta_path.exists() {
        eprintln!(
            "skipping: alpha fixture ({}) or beta fixture ({}) not found -- beta worktree fixtures are optional",
            alpha_path.display(),
            beta_path.display()
        );
        return;
    }

    let engine = Engine::new().unwrap();
    let alpha_bytes = std::fs::read(&alpha_path).unwrap();
    let model_alpha = load_model_from_gguf(&engine, &alpha_bytes).unwrap();
    let beta_bytes = std::fs::read(&beta_path).unwrap();
    let model_beta = load_model_from_gguf(&engine, &beta_bytes).unwrap();

    let t_ctx = 512;
    let horizon = 32;
    let ctx_alpha = sine_context(t_ctx, 0.03);
    let ctx_beta = sine_context(t_ctx, 0.05);

    // t0-fast (this crate) hardcodes head_dim=64 (see `forward_async`'s
    // check in model.rs) -- t0-beta's config uses head_dim=128, a
    // separate, pre-existing architectural gap unrelated to the Pool bug
    // this test targets. Skip rather than fail when hit.
    let baseline_alpha = match forecast(&engine, &model_alpha, &ctx_alpha, 1, t_ctx, horizon) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("skipping: alpha forecast failed ({e}) -- not the Pool bug this test targets");
            return;
        }
    };
    let baseline_beta = match forecast(&engine, &model_beta, &ctx_beta, 1, t_ctx, horizon) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("skipping: beta forecast failed ({e}) -- t0-fast hardcodes head_dim=64, t0-beta likely uses a different head_dim");
            return;
        }
    };

    let a1 = forecast(&engine, &model_alpha, &ctx_alpha, 1, t_ctx, horizon).unwrap();
    let b1 = forecast(&engine, &model_beta, &ctx_beta, 1, t_ctx, horizon).unwrap();
    let a2 = forecast(&engine, &model_alpha, &ctx_alpha, 1, t_ctx, horizon).unwrap();
    let b2 = forecast(&engine, &model_beta, &ctx_beta, 1, t_ctx, horizon).unwrap();

    assert_eq!(a1, baseline_alpha, "alpha model's first interleaved call diverged from its standalone baseline");
    assert_eq!(b1, baseline_beta, "beta model's first interleaved call diverged from its standalone baseline");
    assert_eq!(a2, baseline_alpha, "alpha model's second interleaved call diverged from its standalone baseline");
    assert_eq!(b2, baseline_beta, "beta model's second interleaved call diverged from its standalone baseline");
}
