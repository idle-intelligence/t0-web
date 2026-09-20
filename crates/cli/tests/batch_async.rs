//! `forecast_batch_chunked_async` (the `into_data_async().await` twin of
//! `forecast_batch_chunked`, added for `T0Wasm::forecastBatch` — see
//! `docs/runs/2026-09-20-head-to-head.md`) must produce exactly the same
//! output as the sync version it mirrors. `#[ignore]`d: needs the t0-alpha
//! checkpoint (never committed).
#![cfg(feature = "ndarray")]

use std::path::PathBuf;

use burn_ndarray::NdArray;
use t0_core::{forecast_batch_chunked, forecast_batch_chunked_async, T0Config, T0Model, Weights};

type B = NdArray<f32>;

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
#[ignore = "requires the t0-alpha checkpoint; run `hf download theforecastingcompany/t0-alpha --local-dir ~/Code/idle-intelligence/models/hf/theforecastingcompany/t0-alpha` first, then `cargo test -p t0-cli --release -- --ignored batch_async`"]
fn async_chunked_batch_matches_sync_chunked_batch() {
    let home = std::env::var("HOME").unwrap();
    let model_dir = PathBuf::from(home).join("Code/idle-intelligence/models/hf/theforecastingcompany/t0-alpha");
    assert!(model_dir.join("model.safetensors").exists(), "{} not found — see the #[ignore] reason above", model_dir.display());

    let config: T0Config = serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json")).unwrap()).unwrap();
    let weights = Weights::load(&model_dir.join("model.safetensors")).unwrap();
    let device = Default::default();
    let model = T0Model::<B>::load(&weights, config, &device).unwrap();

    let t_ctx = 512;
    let horizon = 32;
    let n = 24;
    let contexts = synthetic_sines(n, t_ctx);

    let sync_out = forecast_batch_chunked(&model, &contexts, n, t_ctx, horizon, &device, 16);
    let async_out = pollster::block_on(forecast_batch_chunked_async(&model, &contexts, n, t_ctx, horizon, &device, 16));

    assert_eq!(sync_out.len(), async_out.len());
    let max_abs = sync_out.iter().zip(&async_out).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_abs <= 1e-6, "async batch diverges from sync batch by {max_abs:.3e}");
}
