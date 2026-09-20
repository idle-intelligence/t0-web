//! `t0-cli`: native driver for `t0-core`. Backend is picked by Cargo
//! feature (`ndarray` default, `wgpu` optional) — never inside the library
//! crate, per this repo's CLAUDE.md. `--backend` is asserted against the
//! compiled feature (not a runtime switch: two different `Backend` types
//! can't coexist in one binary without dynamic dispatch, and RAM is tight
//! enough here that we build one backend at a time anyway).

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use t0_core::weights::Quant;
use t0_core::{forecast, forecast_batch_chunked, forecast_rollout, T0Config, T0Model, Weights, DEFAULT_BATCH_CHUNK};

#[cfg(feature = "ndarray")]
type Backend = burn_ndarray::NdArray<f32>;
#[cfg(all(feature = "wgpu", not(feature = "ndarray")))]
type Backend = burn_wgpu::Wgpu<f32, i32>;

#[cfg(feature = "ndarray")]
const BACKEND_NAME: &str = "ndarray";
#[cfg(all(feature = "wgpu", not(feature = "ndarray")))]
const BACKEND_NAME: &str = "wgpu";

fn device() -> <Backend as burn::tensor::backend::Backend>::Device {
    Default::default()
}

fn check_backend_flag(requested: &str) -> Result<()> {
    if requested == "fast" {
        return Err(anyhow!("--backend fast is not a Burn backend; this codepath should have dispatched to the t0-fast commands directly"));
    }
    if requested != BACKEND_NAME {
        return Err(anyhow!(
            "--backend {requested} requested but this binary was built with --features {BACKEND_NAME} \
             (backend is a compile-time Cargo feature, not a runtime switch — rebuild with --features {requested})"
        ));
    }
    Ok(())
}

#[derive(Deserialize)]
struct Manifest {
    context_len: usize,
    horizon: usize,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct GiftWindow {
    context_len: usize,
    context_file: String,
}

#[derive(Deserialize)]
struct GiftTask {
    config: String,
    horizon: usize,
    windows: Vec<GiftWindow>,
}

#[derive(Deserialize)]
struct GiftManifest {
    tasks: Vec<GiftTask>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    v: usize,
    n_patches: usize,
    context_file: String,
    quantiles_file: String,
    patch_embedding_file: String,
    layer0_output_file: String,
}

/// `tools/make_fixtures.py`'s `manifest_long.json`: same idea as
/// `Manifest`/`Case`, but each case carries its own `context_len`/`horizon`
/// (rollout fixtures aren't all the same size the way the milestone-1 ones
/// are) and skips the patch-embedding/layer0 trace comparison -- a rollout
/// runs `model.forward` several times over different windows, so "the"
/// trace isn't a single well-defined thing to compare.
#[derive(Deserialize)]
struct LongManifest {
    quantile_levels: Vec<f32>,
    cases: Vec<LongCase>,
}

#[derive(Deserialize)]
struct LongCase {
    name: String,
    v: usize,
    context_len: usize,
    horizon: usize,
    context_file: String,
    quantiles_file: String,
}

fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

fn max_abs_err(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let abs = (x - y).abs();
        max_abs = max_abs.max(abs);
        let rel = abs / y.abs().max(1e-6);
        max_rel = max_rel.max(rel);
    }
    (max_abs, max_rel)
}

fn load_model(weights_path: &Path, config_path: Option<&Path>) -> Result<(T0Model<Backend>, std::time::Duration)> {
    let t0 = Instant::now();
    let (weights, config) = Weights::load_auto(weights_path, config_path)?;
    let model = T0Model::load(&weights, config, &device())?;
    Ok((model, t0.elapsed()))
}

fn cmd_forecast(fixtures_dir: &Path, weights: &Path, config: Option<&Path>, case_idx: usize) -> Result<()> {
    let manifest: Manifest = serde_json::from_str(&std::fs::read_to_string(fixtures_dir.join("manifest.json"))?)?;
    let case = manifest.cases.get(case_idx).ok_or_else(|| anyhow!("no fixture #{case_idx}"))?;
    let context = read_f32(&fixtures_dir.join(&case.context_file))?;
    let (model, load_time) = load_model(weights, config)?;
    println!("loaded in {:.3}s", load_time.as_secs_f64());
    let (out, _) = forecast(&model, &context, case.v, manifest.context_len, manifest.horizon, &device(), false);
    println!("forecast {} ({} values): {:?}...", case.name, out.len(), &out[..out.len().min(10)]);
    Ok(())
}

fn cmd_parity(fixtures_dir: &Path, weights: &Path, config: Option<&Path>) -> Result<()> {
    let manifest: Manifest = serde_json::from_str(&std::fs::read_to_string(fixtures_dir.join("manifest.json"))?)?;
    let (model, load_time) = load_model(weights, config)?;
    println!("loaded in {:.3}s", load_time.as_secs_f64());
    let dev = device();
    println!("{:<28} {:>14} {:>14} {:>14} {:>14}", "case", "quant_max_abs", "quant_max_rel", "patch_max_abs", "layer0_max_abs");
    let mut worst = 0.0f32;
    for case in &manifest.cases {
        let context = read_f32(&fixtures_dir.join(&case.context_file))?;
        let expected_q = read_f32(&fixtures_dir.join(&case.quantiles_file))?;
        let expected_pe = read_f32(&fixtures_dir.join(&case.patch_embedding_file))?;
        let expected_l0 = read_f32(&fixtures_dir.join(&case.layer0_output_file))?;

        let (got_q, trace) = forecast(&model, &context, case.v, manifest.context_len, manifest.horizon, &dev, true);
        let trace = trace.expect("want_trace was true");
        let got_pe: Vec<f32> = trace.patch_embedding.into_data().iter::<f32>().collect();
        let got_l0: Vec<f32> = trace.layer0_output.into_data().iter::<f32>().collect();

        let (q_abs, q_rel) = max_abs_err(&got_q, &expected_q);
        let (pe_abs, _) = max_abs_err(&got_pe, &expected_pe);
        let (l0_abs, _) = max_abs_err(&got_l0, &expected_l0);
        println!("{:<28} {:>14.6e} {:>14.6e} {:>14.6e} {:>14.6e}", case.name, q_abs, q_rel, pe_abs, l0_abs);
        worst = worst.max(q_abs);
        let _ = case.n_patches;
    }
    println!("worst quantile max-abs error across fixtures: {worst:.6e}");
    if worst > 1e-4 {
        return Err(anyhow!("parity gate failed: max-abs error {worst:.6e} exceeds 1e-4"));
    }
    println!("parity gate PASSED (<= 1e-4 max-abs)");
    Ok(())
}

/// `parity --long`: same idea as `cmd_parity`, but against
/// `manifest_long.json`'s rollout fixture(s) via `forecast_rollout` instead
/// of the single-window `forecast`. `v` must be 1 -- `forecast_rollout` is
/// restricted to one univariate target series (see its doc comment).
fn cmd_parity_long(fixtures_dir: &Path, weights: &Path, config: Option<&Path>) -> Result<()> {
    let manifest: LongManifest = serde_json::from_str(&std::fs::read_to_string(fixtures_dir.join("manifest_long.json"))?)?;
    let (model, load_time) = load_model(weights, config)?;
    println!("loaded in {:.3}s", load_time.as_secs_f64());
    let dev = device();
    println!("{:<28} {:>10} {:>10} {:>14} {:>14}", "case", "context", "horizon", "quant_max_abs", "quant_max_rel");
    let mut worst = 0.0f32;
    for case in &manifest.cases {
        assert_eq!(case.v, 1, "forecast_rollout only supports a single univariate target series");
        let context = read_f32(&fixtures_dir.join(&case.context_file))?;
        let expected_q = read_f32(&fixtures_dir.join(&case.quantiles_file))?;
        let got_q = forecast_rollout(&model, &context, case.context_len, case.horizon, &manifest.quantile_levels, &dev);
        let (q_abs, q_rel) = max_abs_err(&got_q, &expected_q);
        println!("{:<28} {:>10} {:>10} {:>14.6e} {:>14.6e}", case.name, case.context_len, case.horizon, q_abs, q_rel);
        worst = worst.max(q_abs);
    }
    println!("worst quantile max-abs error across long fixtures: {worst:.6e}");
    if worst > 1e-4 {
        return Err(anyhow!("long parity gate failed: max-abs error {worst:.6e} exceeds 1e-4"));
    }
    println!("long parity gate PASSED (<= 1e-4 max-abs)");
    Ok(())
}

/// Forecasts every window of every task in a GIFT-Eval subset manifest
/// (`tools/gifteval_subset.py`'s output) and writes one little-endian f32
/// file per task: `n_windows * horizon * n_quantiles` values, window-major,
/// then time-major, then quantile-major (quantile order == the model's
/// trained `quantile_levels`, e.g. [0.1, 0.25, 0.5, 0.75, 0.9] for
/// t0-alpha). Scoring (CRPS/MASE via gluonts, plus interpolating the
/// trained quantiles up to GIFT-Eval's 9 query levels) happens in Python —
/// see `tools/score_gifteval.py` — so the metric matches the official
/// harness exactly, not a Rust reimplementation of it.
fn cmd_gifteval(manifest_dir: &Path, weights: &Path, config: Option<&Path>, out_dir: &Path) -> Result<()> {
    let manifest: GiftManifest = serde_json::from_str(&std::fs::read_to_string(manifest_dir.join("manifest.json"))?)?;
    let (model, load_time) = load_model(weights, config)?;
    println!("loaded in {:.3}s", load_time.as_secs_f64());
    let dev = device();
    std::fs::create_dir_all(out_dir)?;
    let n_q = model.config.n_quantiles();

    for task in &manifest.tasks {
        let safe_name = task.config.replace('/', "_");
        let mut out = Vec::with_capacity(task.windows.len() * task.horizon * n_q);
        let t0 = Instant::now();
        for window in &task.windows {
            let context = read_f32(&manifest_dir.join(&window.context_file))?;
            assert_eq!(context.len(), window.context_len);
            let (q, _) = forecast(&model, &context, 1, window.context_len, task.horizon, &dev, false);
            out.extend_from_slice(&q);
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let out_path = out_dir.join(format!("{safe_name}.f32"));
        let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&out_path, &bytes)?;
        println!(
            "{}: {} windows forecast in {:.2}s ({:.1} ms/window) -> {}",
            task.config,
            task.windows.len(),
            elapsed,
            1000.0 * elapsed / task.windows.len().max(1) as f64,
            out_path.display()
        );
    }
    Ok(())
}

/// Forecasts one window sliced directly out of a raw little-endian f32
/// series file (e.g. `web/data/series.f32`) and writes the raw
/// `horizon * n_quantiles` output to `out` (same layout as `gifteval`'s
/// per-window output). Exists to produce an F32 reference for comparing
/// against a third-party export (e.g. the official ONNX INT8 build) on the
/// exact same context/horizon window this repo's own demo page uses — see
/// `docs/runs/2026-09-20-head-to-head.md`.
#[allow(clippy::too_many_arguments)]
fn cmd_forecast_raw(
    weights_path: &Path,
    config_path: Option<&Path>,
    series_path: &Path,
    origin: usize,
    t_ctx: usize,
    horizon: usize,
    rollout: bool,
    out: &Path,
) -> Result<()> {
    let series = read_f32(series_path)?;
    let ctx_start = origin.saturating_sub(t_ctx);
    let context = &series[ctx_start..origin];
    let (model, load_time) = load_model(weights_path, config_path)?;
    let q = if rollout {
        let dev = device();
        let query_levels = model.config.quantile_levels.clone();
        forecast_rollout(&model, context, context.len(), horizon, &query_levels, &dev)
    } else {
        forecast(&model, context, 1, context.len(), horizon, &device(), false).0
    };
    let bytes: Vec<u8> = q.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(out, &bytes)?;
    println!(
        "loaded in {:.3}s, context_len={}, wrote {} values ({} bytes) to {}",
        load_time.as_secs_f64(),
        context.len(),
        q.len(),
        bytes.len(),
        out.display()
    );
    Ok(())
}

fn cmd_export_gguf(weights_path: &Path, config_path: &Path, quant: &str, out: &Path) -> Result<()> {
    let quant = Quant::parse(quant)?;
    let config: t0_core::T0Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
    let weights = Weights::load(weights_path)?;
    weights.export_gguf(&config, quant, out)?;

    let in_size = std::fs::metadata(weights_path)?.len();
    let out_size = std::fs::metadata(out)?.len();
    println!("{}: {} bytes ({:.1} MB)", weights_path.display(), in_size, in_size as f64 / 1e6);
    println!("{}: {} bytes ({:.1} MB)", out.display(), out_size, out_size as f64 / 1e6);
    println!("ratio: {:.3}x", out_size as f64 / in_size as f64);
    Ok(())
}

/// Synthetic sine signals: `n_signals` rows of `t_ctx` samples, distinct
/// frequency per row so they're not bit-identical (matters for cache/branch
/// timing, not for correctness).
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

/// xorshift64* — no `rand` dependency needed for a small deterministic
/// case generator.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Box-Muller, standard normal.
    fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

/// Synthetic univariate drift-benchmark case: a sine (+ optional second
/// harmonic and linear trend) plus Gaussian noise, deterministic given
/// `(seed, idx)`. **Assumption** (documented in `docs/BENCHMARKS.md`): the
/// HF INT8 cards state the drift protocol ("54 synthetic cases", the drift
/// formula) but publish no case generator, so this is our own
/// reconstruction, not a verified reproduction of theirs. Every 4th case
/// gets a contiguous NaN gap (mirrors `fixtures/case2_masked_gap`) so the
/// scaler's missing-value path is exercised too.
fn synthetic_case(seed: u64, idx: usize, t_ctx: usize) -> Vec<f32> {
    let mut rng = Rng(seed.wrapping_add(idx as u64).wrapping_mul(0x9E3779B97F4A7C15) | 1);
    let freq1 = 0.01 + 0.04 * rng.uniform();
    let amp1 = 0.5 + 2.0 * rng.uniform();
    let phase1 = 2.0 * std::f32::consts::PI * rng.uniform();
    let freq2 = 0.05 + 0.4 * rng.uniform();
    let amp2 = amp1 * 0.3 * rng.uniform();
    let trend = (rng.uniform() - 0.5) * 0.01;
    let noise_sd = 0.05 * amp1 * rng.uniform();

    let mut out = vec![0.0f32; t_ctx];
    for (t, v) in out.iter_mut().enumerate() {
        let x = t as f32;
        *v = amp1 * (freq1 * x + phase1).sin() + amp2 * (freq2 * x).sin() + trend * x + noise_sd * rng.normal();
    }
    if idx % 4 == 3 {
        let gap_len = t_ctx / 8;
        let gap_start = t_ctx / 2;
        for v in out.iter_mut().skip(gap_start).take(gap_len) {
            *v = f32::NAN;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn cmd_drift(weights_f32_path: &Path, config_path: &Path, quant: &str, n_cases: usize, seed: u64) -> Result<()> {
    let config: T0Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
    let f32_weights = Weights::load(weights_f32_path)?;
    let dev = device();
    let f32_model = T0Model::<Backend>::load(&f32_weights, config.clone(), &dev)?;

    let (quant_weights, quant_config, size_bytes, label): (Weights, T0Config, u64, String) = if quant == "int8-theirs" {
        (
            f32_weights.quantize_int8_theirs(),
            config.clone(),
            f32_weights.estimated_int8_theirs_bytes(),
            "int8-theirs (estimated size)".to_string(),
        )
    } else {
        let path = Path::new(quant);
        let (w, c) = Weights::load_auto(path, None)?;
        (w, c, std::fs::metadata(path)?.len(), quant.to_string())
    };
    let quant_model = T0Model::<Backend>::load(&quant_weights, quant_config, &dev)?;

    let t_ctx = 512usize;
    let horizon = 96usize;

    let mut mean_drifts = Vec::with_capacity(n_cases);
    let mut point_drifts = Vec::with_capacity(n_cases);
    for i in 0..n_cases {
        let context = synthetic_case(seed, i, t_ctx);
        let (reference, _) = forecast(&f32_model, &context, 1, t_ctx, horizon, &dev, false);
        let (quantized, _) = forecast(&quant_model, &context, 1, t_ctx, horizon, &dev, false);
        assert_eq!(reference.len(), quantized.len());

        let range = reference.iter().cloned().fold(f32::MIN, f32::max) - reference.iter().cloned().fold(f32::MAX, f32::min);
        let range = range.max(1e-6);

        let mut mean_drift = 0.0f32;
        let mut point_drift = 0.0f32;
        for (r, q) in reference.iter().zip(&quantized) {
            let drift = (r - q).abs() / range;
            mean_drift += drift;
            point_drift = point_drift.max(drift);
        }
        mean_drift /= reference.len() as f32;
        mean_drifts.push(mean_drift);
        point_drifts.push(point_drift);
    }

    let mean_worst = mean_drifts.iter().cloned().fold(0.0f32, f32::max);
    let mean_mean = mean_drifts.iter().sum::<f32>() / n_cases as f32;
    let point_worst = point_drifts.iter().cloned().fold(0.0f32, f32::max);
    let point_mean = point_drifts.iter().sum::<f32>() / n_cases as f32;

    println!(
        "quant={label} cases={n_cases} seed={seed} file_bytes={size_bytes} file_mb={:.1} \
         mean_drift_worst_pct={:.4} mean_drift_mean_pct={:.4} point_drift_worst_pct={:.4} point_drift_mean_pct={:.4}",
        size_bytes as f64 / 1e6,
        mean_worst * 100.0,
        mean_mean * 100.0,
        point_worst * 100.0,
        point_mean * 100.0,
    );
    Ok(())
}

/// `t0-fast` (no Burn at inference) parity check: same fixtures, same
/// 1e-4 max-abs gate as `cmd_parity`, but only the final quantile output
/// Sum of `out_dim*in_dim*4` bytes for the four big per-layer matmuls
/// (`wQKV`, `wO`, `mlp.0`, `mlp.2`) if resident as F32 -- the "before" side
/// of `t0-fast`'s Phase B memory report; the "after" side is
/// `GpuModel::quantizable_gpu_bytes()`.
#[cfg(feature = "fast")]
fn quantizable_f32_bytes(weights: &Weights, config: &T0Config) -> Result<u64> {
    let mut total = 0u64;
    for i in 0..config.num_layers {
        let p = format!("transformer.layers.{i}");
        for name in [
            format!("{p}.attention_block.attention.wQKV.weight"),
            format!("{p}.attention_block.attention.wO.weight"),
            format!("{p}.mlp.0.weight"),
            format!("{p}.mlp.2.weight"),
        ] {
            let shape = weights.shape(&name)?;
            total += (shape[0] * shape[1] * 4) as u64;
        }
    }
    Ok(total)
}

/// Same data path as `cmd_gifteval` (same manifest, same per-window
/// forecast-and-write loop), driven by `t0_fast::forecast` instead of a
/// Burn `T0Model`. `weights_path` picks the load path exactly like
/// `cmd_parity_fast`: a `.gguf` file loads GGUF-resident (Q8_0/Q4_0,
/// whatever quant the file was exported at); a `.safetensors` file loads
/// through `t0_fast::load_model` with `fast_quant` (used for the F32
/// residency, since `export-gguf` has no F32 quant scheme). Every window
/// here goes through the single-window `forecast` (no rollout) — valid as
/// long as every task's horizon fits in one forward pass
/// (`horizon <= t0_core::MAX_HORIZON`, true for every GIFT-Eval config used
/// in this repo, capped or official-protocol).
#[cfg(feature = "fast")]
fn cmd_gifteval_fast(manifest_dir: &Path, weights_path: &Path, config_path: Option<&Path>, out_dir: &Path, fast_quant: &str) -> Result<()> {
    let manifest: GiftManifest = serde_json::from_str(&std::fs::read_to_string(manifest_dir.join("manifest.json"))?)?;
    let engine = t0_fast::Engine::new()?;
    let is_gguf = weights_path.extension().and_then(|e| e.to_str()) == Some("gguf");
    let t0 = Instant::now();
    let model = if is_gguf {
        let bytes = std::fs::read(weights_path)?;
        t0_fast::load_model_from_gguf(&engine, &bytes)?
    } else {
        let (weights, config) = Weights::load_auto(weights_path, config_path)?;
        let quant = t0_fast::WeightQuant::parse(fast_quant)?;
        t0_fast::load_model(&engine, &weights, config, quant)?
    };
    println!("loaded in {:.3}s", t0.elapsed().as_secs_f64());
    std::fs::create_dir_all(out_dir)?;
    let n_q = model.config.n_quantiles();

    for task in &manifest.tasks {
        let safe_name = task.config.replace('/', "_");
        let mut out = Vec::with_capacity(task.windows.len() * task.horizon * n_q);
        let t0 = Instant::now();
        for window in &task.windows {
            let context = read_f32(&manifest_dir.join(&window.context_file))?;
            assert_eq!(context.len(), window.context_len);
            let q = t0_fast::forecast(&engine, &model, &context, 1, window.context_len, task.horizon)?;
            out.extend_from_slice(&q);
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let out_path = out_dir.join(format!("{safe_name}.f32"));
        let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&out_path, &bytes)?;
        println!(
            "{}: {} windows forecast in {:.2}s ({:.1} ms/window) -> {}",
            task.config,
            task.windows.len(),
            elapsed,
            1000.0 * elapsed / task.windows.len().max(1) as f64,
            out_path.display()
        );
    }
    Ok(())
}

/// F32 (`fast_quant=f32`) compares against the fixtures' F32-reference
/// quantile files, same gate as `cmd_parity`. Q8_0/Q4_0 compare against
/// *Burn's own* Q8_0/Q4_0 path instead (per this task's brief): the F32
/// weights are round-tripped through `Weights::export_gguf`/`load_gguf`
/// (the same `quantize_q8_0`/`quantize_q4_0` math `t0-fast::quant` uses)
/// and forecast on the `ndarray` backend, since the F32 fixtures already
/// bake in quantization error that isn't a `t0-fast` bug.
#[cfg(feature = "fast")]
fn cmd_parity_fast(fixtures_dir: &Path, weights_path: &Path, config_path: Option<&Path>, fast_quant: &str) -> Result<()> {
    let manifest: Manifest = serde_json::from_str(&std::fs::read_to_string(fixtures_dir.join("manifest.json"))?)?;
    let is_gguf = weights_path.extension().and_then(|e| e.to_str()) == Some("gguf");
    let engine = t0_fast::Engine::new()?;

    // Two distinct load paths: a `.gguf` weights file goes straight through
    // `load_model_from_gguf` (block bytes -> GPU buffer, no f32-expanded
    // copy anywhere -- the browser's actual load path per
    // docs/runs/2026-09-20-perf.md's Phase B-web item), everything else
    // (safetensors) goes through the existing f32-in-memory + on-the-fly
    // requantize path this command already had.
    let (weights_opt, config, model) = if is_gguf {
        let bytes = std::fs::read(weights_path)?;
        let model = t0_fast::load_model_from_gguf(&engine, &bytes)?;
        let config = model.config.clone();
        (None, config, model)
    } else {
        let (weights, config) = Weights::load_auto(weights_path, config_path)?;
        let quant = t0_fast::WeightQuant::parse(fast_quant)?;
        let model = t0_fast::load_model(&engine, &weights, config.clone(), quant)?;
        (Some(weights), config, model)
    };
    if let Some(weights) = &weights_opt {
        let f32_bytes = quantizable_f32_bytes(weights, &config)?;
        println!(
            "fast_quant={fast_quant} quantizable_f32_bytes={f32_bytes} quantizable_gpu_bytes={} ratio={:.3}",
            model.quantizable_gpu_bytes(),
            model.quantizable_gpu_bytes() as f64 / f32_bytes as f64
        );
    } else {
        println!("gguf_resident=true quantizable_gpu_bytes={}", model.quantizable_gpu_bytes());
    }

    #[cfg(feature = "ndarray")]
    let burn_reference: Option<T0Model<Backend>> = if fast_quant == "f32" && !is_gguf {
        None
    } else if is_gguf {
        let (roundtripped, rt_config) = Weights::load_gguf(weights_path)?;
        Some(T0Model::load(&roundtripped, rt_config, &device())?)
    } else {
        let weights = weights_opt.as_ref().expect("safetensors path always has weights");
        let core_quant = Quant::parse(fast_quant)?;
        let gguf_path = std::env::temp_dir().join(format!("t0_fast_parity_{fast_quant}.gguf"));
        weights.export_gguf(&config, core_quant, &gguf_path)?;
        let (roundtripped, rt_config) = Weights::load_gguf(&gguf_path)?;
        Some(T0Model::load(&roundtripped, rt_config, &device())?)
    };
    #[cfg(not(feature = "ndarray"))]
    let burn_reference: Option<()> = None;

    println!("{:<28} {:>14}", "case", "quant_max_abs");
    let mut worst = 0.0f32;
    for case in &manifest.cases {
        let context = read_f32(&fixtures_dir.join(&case.context_file))?;
        let got_q = t0_fast::forecast(&engine, &model, &context, case.v, manifest.context_len, manifest.horizon)?;
        #[cfg(feature = "ndarray")]
        let expected_q = match &burn_reference {
            Some(ref_model) => forecast(ref_model, &context, case.v, manifest.context_len, manifest.horizon, &device(), false).0,
            None => read_f32(&fixtures_dir.join(&case.quantiles_file))?,
        };
        #[cfg(not(feature = "ndarray"))]
        let expected_q = read_f32(&fixtures_dir.join(&case.quantiles_file))?;
        let (q_abs, _) = max_abs_err(&got_q, &expected_q);
        println!("{:<28} {:>14.6e}", case.name, q_abs);
        worst = worst.max(q_abs);
    }
    println!("worst quantile max-abs error across fixtures: {worst:.6e}");
    if fast_quant == "f32" && !is_gguf {
        if worst > 1e-4 {
            return Err(anyhow!("parity gate failed: max-abs error {worst:.6e} exceeds 1e-4"));
        }
        println!("parity gate PASSED (<= 1e-4 max-abs)");
    } else {
        // 1e-4 is the "same computation, no Burn" gate (see fast_quant=f32
        // above and Phase A) -- it doesn't apply to a genuinely lossy
        // quantization compared against Burn's own path on the same
        // quantized weights. Report only; see t0-fast::model's Q8_0 kernel
        // unit test for the isolated-kernel-correctness check instead.
        println!("(informational vs Burn's own {fast_quant} path -- not gated at 1e-4, see t0-fast::model tests for kernel correctness)");
    }
    Ok(())
}

/// `parity --long --backend fast`: same idea as `cmd_parity_long`, but
/// through `t0_fast::forecast_rollout` (F32 weights only -- there's no
/// reference to compare a quantized rollout against here, same reasoning as
/// `cmd_parity_fast`'s `fast_quant=f32` gate).
#[cfg(feature = "fast")]
fn cmd_parity_long_fast(fixtures_dir: &Path, weights_path: &Path, config_path: Option<&Path>) -> Result<()> {
    let manifest: LongManifest = serde_json::from_str(&std::fs::read_to_string(fixtures_dir.join("manifest_long.json"))?)?;
    let engine = t0_fast::Engine::new()?;
    let (weights, config) = Weights::load_auto(weights_path, config_path)?;
    let model = t0_fast::load_model(&engine, &weights, config, t0_fast::WeightQuant::F32)?;

    println!("{:<28} {:>10} {:>10} {:>14} {:>14}", "case", "context", "horizon", "quant_max_abs", "quant_max_rel");
    let mut worst = 0.0f32;
    for case in &manifest.cases {
        assert_eq!(case.v, 1, "forecast_rollout only supports a single univariate target series");
        let context = read_f32(&fixtures_dir.join(&case.context_file))?;
        let expected_q = read_f32(&fixtures_dir.join(&case.quantiles_file))?;
        let got_q = t0_fast::forecast_rollout(&engine, &model, &context, case.context_len, case.horizon, &manifest.quantile_levels)?;
        let (q_abs, q_rel) = max_abs_err(&got_q, &expected_q);
        println!("{:<28} {:>10} {:>10} {:>14.6e} {:>14.6e}", case.name, case.context_len, case.horizon, q_abs, q_rel);
        worst = worst.max(q_abs);
    }
    println!("worst quantile max-abs error across long fixtures (fast backend): {worst:.6e}");
    if worst > 1e-4 {
        return Err(anyhow!("long parity gate failed: max-abs error {worst:.6e} exceeds 1e-4"));
    }
    println!("long parity gate PASSED (<= 1e-4 max-abs)");
    Ok(())
}

#[cfg(feature = "fast")]
#[allow(clippy::too_many_arguments)]
fn cmd_bench_fast(weights_path: &Path, config_path: Option<&Path>, n_signals: usize, t_ctx: usize, horizon: usize, reps: usize, warmup: usize, fast_quant: &str, chunk: usize) -> Result<()> {
    let file_size = std::fs::metadata(weights_path)?.len();
    let is_gguf = weights_path.extension().and_then(|e| e.to_str()) == Some("gguf");
    let engine = t0_fast::Engine::new()?;
    let t0 = Instant::now();
    let model = if is_gguf {
        let bytes = std::fs::read(weights_path)?;
        t0_fast::load_model_from_gguf(&engine, &bytes)?
    } else {
        let quant = t0_fast::WeightQuant::parse(fast_quant)?;
        let (weights, config) = Weights::load_auto(weights_path, config_path)?;
        t0_fast::load_model(&engine, &weights, config, quant)?
    };
    let load_time = t0.elapsed();
    let quantizable_gpu_bytes = model.quantizable_gpu_bytes();
    let context = synthetic_sines(n_signals, t_ctx);
    // n_signals independent forecasts (t0_core::forecast_batch_chunked's
    // TimeSeries::from_context_batch semantics), chunked exactly like
    // cmd_bench's Burn path -- see t0-fast::forecast_batch_chunked.
    let run = || t0_fast::forecast_batch_chunked(&engine, &model, &context, n_signals, t_ctx, horizon, chunk);

    for _ in 0..warmup {
        let _ = run()?;
    }
    engine.pool.reset_alloc_count();
    let _ = run()?;
    let warm_alloc_count = engine.pool.alloc_count();

    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        let _ = run()?;
        times.push(t0.elapsed().as_secs_f64());
    }
    let med = median(times.clone());
    println!(
        "backend=fast fast_quant={fast_quant} weights={} file_bytes={file_size} load_s={:.3} n_signals={n_signals} t_ctx={t_ctx} horizon={horizon} \
         chunk={chunk} reps={reps} median_s={med:.4} per_signal_ms={:.4} warm_alloc_count={warm_alloc_count} quantizable_gpu_bytes={quantizable_gpu_bytes} all_s={times:?}",
        weights_path.display(),
        load_time.as_secs_f64(),
        (med * 1000.0) / n_signals as f64,
    );
    Ok(())
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

#[allow(clippy::too_many_arguments)]
fn cmd_bench(weights_path: &Path, config_path: Option<&Path>, n_signals: usize, t_ctx: usize, horizon: usize, reps: usize, chunk: usize, warmup: usize) -> Result<()> {
    let file_size = std::fs::metadata(weights_path)?.len();
    let (model, load_time) = load_model(weights_path, config_path)?;
    let dev = device();
    let context = synthetic_sines(n_signals, t_ctx);

    // `warmup` untimed passes, then `reps` timed passes.
    for _ in 0..warmup {
        let _ = forecast_batch_chunked(&model, &context, n_signals, t_ctx, horizon, &dev, chunk);
    }
    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        let _ = forecast_batch_chunked(&model, &context, n_signals, t_ctx, horizon, &dev, chunk);
        times.push(t0.elapsed().as_secs_f64());
    }
    let med = median(times.clone());
    println!(
        "backend={BACKEND_NAME} weights={} file_bytes={file_size} load_s={:.3} n_signals={n_signals} t_ctx={t_ctx} horizon={horizon} \
         chunk={chunk} reps={reps} median_s={med:.4} per_signal_ms={:.4} all_s={times:?} threads_available={}",
        weights_path.display(),
        load_time.as_secs_f64(),
        (med * 1000.0) / n_signals as f64,
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut fixtures = PathBuf::from("fixtures");
    let mut weights = PathBuf::from("../../models/hf/theforecastingcompany/t0-alpha/model.safetensors");
    let mut config: Option<PathBuf> = Some(PathBuf::from("../../models/hf/theforecastingcompany/t0-alpha/config.json"));
    let mut fixture_idx = 0usize;
    let mut backend = BACKEND_NAME.to_string();
    let mut quant = "q8_0".to_string();
    #[cfg(feature = "fast")]
    let mut fast_quant = "f32".to_string();
    let mut out = PathBuf::from("out.gguf");
    let mut n_signals = 1usize;
    let mut t_ctx = 512usize;
    let mut horizon = 96usize;
    let mut reps = 5usize;
    let mut chunk = DEFAULT_BATCH_CHUNK;
    let mut warmup = 1usize;
    let mut weights_f32: Option<PathBuf> = None;
    let mut cases = 54usize;
    let mut seed = 0u64;
    let mut manifest_dir = PathBuf::from("fixtures/gifteval");
    let mut gift_out = PathBuf::from("fixtures/gifteval/forecasts");
    let mut series_file: Option<PathBuf> = None;
    let mut origin = 0usize;
    let mut long = false;
    let mut rollout = false;
    let cmd = args.get(1).cloned().unwrap_or_default();

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--fixtures" => {
                fixtures = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--weights" => {
                weights = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--config" => {
                config = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--fixture" => {
                fixture_idx = args[i + 1].parse()?;
                i += 2;
            }
            "--backend" => {
                backend = args[i + 1].clone();
                i += 2;
            }
            "--quant" => {
                quant = args[i + 1].clone();
                i += 2;
            }
            "--fast-quant" => {
                #[cfg(feature = "fast")]
                {
                    fast_quant = args[i + 1].clone();
                }
                i += 2;
            }
            "--out" => {
                out = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--signals" => {
                n_signals = args[i + 1].parse()?;
                i += 2;
            }
            "--context" => {
                t_ctx = args[i + 1].parse()?;
                i += 2;
            }
            "--horizon" => {
                horizon = args[i + 1].parse()?;
                i += 2;
            }
            "--reps" => {
                reps = args[i + 1].parse()?;
                i += 2;
            }
            "--chunk" => {
                chunk = args[i + 1].parse()?;
                i += 2;
            }
            "--warmup" => {
                warmup = args[i + 1].parse()?;
                i += 2;
            }
            "--weights-f32" => {
                weights_f32 = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--cases" => {
                cases = args[i + 1].parse()?;
                i += 2;
            }
            "--seed" => {
                seed = args[i + 1].parse()?;
                i += 2;
            }
            "--manifest" => {
                manifest_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--out-dir" => {
                gift_out = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--series-file" => {
                series_file = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--origin" => {
                origin = args[i + 1].parse()?;
                i += 2;
            }
            "--long" => {
                long = true;
                i += 1;
            }
            "--rollout" => {
                rollout = true;
                i += 1;
            }
            other => return Err(anyhow!("unknown argument: {other}")),
        }
    }

    match cmd.as_str() {
        "forecast" => {
            check_backend_flag(&backend)?;
            cmd_forecast(&fixtures, &weights, config.as_deref(), fixture_idx)
        }
        "parity" => {
            #[cfg(feature = "fast")]
            if backend == "fast" {
                return if long {
                    cmd_parity_long_fast(&fixtures, &weights, config.as_deref())
                } else {
                    cmd_parity_fast(&fixtures, &weights, config.as_deref(), &fast_quant)
                };
            }
            check_backend_flag(&backend)?;
            if long {
                cmd_parity_long(&fixtures, &weights, config.as_deref())
            } else {
                cmd_parity(&fixtures, &weights, config.as_deref())
            }
        }
        "export-gguf" => cmd_export_gguf(&weights, config.as_deref().ok_or_else(|| anyhow!("--config is required"))?, &quant, &out),
        "bench" => {
            #[cfg(feature = "fast")]
            if backend == "fast" {
                return cmd_bench_fast(&weights, config.as_deref(), n_signals, t_ctx, horizon, reps, warmup, &fast_quant, chunk);
            }
            check_backend_flag(&backend)?;
            cmd_bench(&weights, config.as_deref(), n_signals, t_ctx, horizon, reps, chunk, warmup)
        }
        "gifteval" => {
            #[cfg(feature = "fast")]
            if backend == "fast" {
                return cmd_gifteval_fast(&manifest_dir, &weights, config.as_deref(), &gift_out, &fast_quant);
            }
            cmd_gifteval(&manifest_dir, &weights, config.as_deref(), &gift_out)
        }
        "forecast-raw" => {
            check_backend_flag(&backend)?;
            let series_file = series_file.ok_or_else(|| anyhow!("--series-file is required"))?;
            cmd_forecast_raw(&weights, config.as_deref(), &series_file, origin, t_ctx, horizon, rollout, &out)
        }
        "drift" => {
            let weights_f32 = weights_f32.ok_or_else(|| anyhow!("--weights-f32 is required"))?;
            let config = config.ok_or_else(|| anyhow!("--config is required"))?;
            cmd_drift(&weights_f32, &config, &quant, cases, seed)
        }
        _ => Err(anyhow!(
            "usage: t0-cli <forecast --fixture N|parity|bench --signals N [--chunk N]|export-gguf --quant f16|q8_0|q4_0 --out FILE| \
             drift --weights-f32 PATH --config PATH --quant f16.gguf|q8_0.gguf|q4_0.gguf|int8-theirs [--cases 54] [--seed 0]| \
             gifteval --manifest DIR --weights PATH [--out-dir DIR]> \
             [--fixtures DIR] [--weights PATH] [--config PATH] [--backend ndarray|wgpu]"
        )),
    }
}
