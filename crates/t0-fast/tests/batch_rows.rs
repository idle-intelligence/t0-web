//! `forecast_batch_rows_chunked` (ragged per-row contexts, added for
//! `T0Wasm::forecastBatchRows`) must match a standalone `forecast()` call
//! on each row's own (unpadded) context exactly -- see
//! `forecast_batch_rows_chunked_async`'s doc comment in `src/lib.rs` for
//! why the extra left-padding a shorter row picks up is numerically inert.
//! `#[ignore]`d: needs a GGUF export (never committed).

use std::path::PathBuf;

use t0_fast::{forecast, forecast_batch_rows_chunked, load_model_from_gguf, Engine};

fn synthetic_sines(n_signals: usize, base_len: usize) -> (Vec<f32>, Vec<usize>) {
    // Distinct, non-patch-aligned lengths on purpose: 512, 511, 480 (a
    // whole patch shorter), 500, repeating -- exercises both the
    // within-patch pad_left case and the whole-extra-patch case.
    let deltas: [i64; 4] = [0, -1, -32, -12];
    let mut lengths = Vec::with_capacity(n_signals);
    let mut contexts = Vec::new();
    for row in 0..n_signals {
        let len = (base_len as i64 + deltas[row % deltas.len()]) as usize;
        let freq = 0.02 + 0.001 * (row % 37) as f32;
        for col in 0..len {
            contexts.push((freq * col as f32).sin());
        }
        lengths.push(len);
    }
    (contexts, lengths)
}

#[test]
#[ignore = "requires a GGUF export; run `bench/ours/t0-alpha-q8_0.gguf` first (see docs/runs), then `cargo test -p t0-fast --release -- --ignored batch_rows`"]
fn ragged_batch_rows_match_single_signal_forecasts() {
    let gguf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../bench/ours/t0-alpha-q8_0.gguf");
    assert!(gguf_path.exists(), "{} not found -- see the #[ignore] reason above", gguf_path.display());

    let engine = Engine::new().unwrap();
    let gguf_bytes = std::fs::read(&gguf_path).unwrap();
    let model = load_model_from_gguf(&engine, &gguf_bytes).unwrap();

    let horizon = 32;
    let n = 6;
    let (contexts, lengths) = synthetic_sines(n, 512);
    let offsets: Vec<usize> = {
        let mut acc = 0;
        let mut v = Vec::with_capacity(n + 1);
        for &l in &lengths {
            v.push(acc);
            acc += l;
        }
        v.push(acc);
        v
    };

    let batched = forecast_batch_rows_chunked(&engine, &model, &contexts, &lengths, horizon, 24).unwrap();
    let per_signal = batched.len() / n;

    for i in 0..n {
        let row_ctx = &contexts[offsets[i]..offsets[i + 1]];
        let single = forecast(&engine, &model, row_ctx, 1, lengths[i], horizon).unwrap();
        assert_eq!(single.len(), per_signal);
        let slice = &batched[i * per_signal..(i + 1) * per_signal];
        let max_abs = slice.iter().zip(&single).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_abs <= 1e-4, "row {i} (len {}) diverges from single-signal forecast by {max_abs:.3e}", lengths[i]);
    }
}
