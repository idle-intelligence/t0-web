//! Chunked wgpu batch must match single-signal ndarray forecasts: this is
//! the workaround for the cubek-matmul shared-memory autotune bug
//! (`docs/runs/2026-09-19-milestone1a.md`) — chunking the batch into groups
//! of `DEFAULT_BATCH_CHUNK` (16) signals keeps every wgpu forward pass's
//! group-attention matmul under Metal's 32 KB shared-memory limit while
//! still producing exactly the un-chunked result. `#[ignore]`d: needs the
//! t0-alpha checkpoint (never committed) and the `wgpu` feature.
#![cfg(feature = "wgpu")]

use std::path::PathBuf;

use burn_wgpu::Wgpu;
use t0_core::{forecast_batch_chunked, T0Config, T0Model, Weights};

type B = Wgpu<f32, i32>;

fn synthetic_sines(n_signals: usize, t_ctx: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_signals * t_ctx];
    for row in 0..n_signals {
        let freq = 0.02 + 0.001 * (row % 37) as f32;
        for col in 0..t_ctx {
            out[row * t_ctx + col] = (freq * col as f32).sin();
        }
    }
    out
}

#[test]
#[ignore = "requires the t0-alpha checkpoint and --features wgpu; run `hf download theforecastingcompany/t0-alpha --local-dir ~/Code/idle-intelligence/models/hf/theforecastingcompany/t0-alpha` first, then `cargo test -p t0-cli --release --no-default-features --features wgpu -- --ignored chunk_wgpu`"]
fn chunked_batch_of_40_equals_40_single() {
    let home = std::env::var("HOME").unwrap();
    let model_dir = PathBuf::from(home).join("Code/idle-intelligence/models/hf/theforecastingcompany/t0-alpha");
    assert!(model_dir.join("model.safetensors").exists(), "{} not found — see the #[ignore] reason above", model_dir.display());

    let config: T0Config = serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json")).unwrap()).unwrap();
    let weights = Weights::load(&model_dir.join("model.safetensors")).unwrap();
    let device = Default::default();
    let model = T0Model::<B>::load(&weights, config, &device).unwrap();

    let t_ctx = 512;
    let horizon = 96;
    let n = 40; // > 16 (DEFAULT_BATCH_CHUNK): would trip the shared-memory bug unchunked.
    let contexts = synthetic_sines(n, t_ctx);

    // Un-chunked single-signal forecasts, one at a time (n=1 never trips the bug).
    let mut singles = Vec::with_capacity(n);
    for i in 0..n {
        let ctx = &contexts[i * t_ctx..(i + 1) * t_ctx];
        let out = forecast_batch_chunked(&model, ctx, 1, t_ctx, horizon, &device, 1);
        singles.push(out);
    }

    // Chunked batch of 40 (chunk size 16, the default).
    let batched = forecast_batch_chunked(&model, &contexts, n, t_ctx, horizon, &device, t0_core::DEFAULT_BATCH_CHUNK);

    let per_signal = singles[0].len();
    assert_eq!(batched.len(), n * per_signal);
    for i in 0..n {
        let slice = &batched[i * per_signal..(i + 1) * per_signal];
        let max_abs = slice.iter().zip(&singles[i]).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_abs <= 1e-5, "signal {i} diverges from single-signal forecast by {max_abs:.3e}");
    }
}
