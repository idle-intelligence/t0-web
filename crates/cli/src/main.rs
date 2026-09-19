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
use t0_core::{forecast, forecast_batch, T0Model, Weights};

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
struct Case {
    name: String,
    v: usize,
    n_patches: usize,
    context_file: String,
    quantiles_file: String,
    patch_embedding_file: String,
    layer0_output_file: String,
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

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn cmd_bench(weights_path: &Path, config_path: Option<&Path>, n_signals: usize, t_ctx: usize, horizon: usize, reps: usize) -> Result<()> {
    let file_size = std::fs::metadata(weights_path)?.len();
    let (model, load_time) = load_model(weights_path, config_path)?;
    let dev = device();
    let context = synthetic_sines(n_signals, t_ctx);

    // 1 warm-up pass (not timed), then `reps` timed passes.
    let _ = forecast_batch(&model, &context, n_signals, t_ctx, horizon, &dev);
    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        let _ = forecast_batch(&model, &context, n_signals, t_ctx, horizon, &dev);
        times.push(t0.elapsed().as_secs_f64());
    }
    let med = median(times.clone());
    println!(
        "backend={BACKEND_NAME} weights={} file_bytes={file_size} load_s={:.3} n_signals={n_signals} t_ctx={t_ctx} horizon={horizon} \
         reps={reps} median_s={med:.4} all_s={times:?} threads_available={}",
        weights_path.display(),
        load_time.as_secs_f64(),
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
    let mut out = PathBuf::from("out.gguf");
    let mut n_signals = 1usize;
    let mut t_ctx = 512usize;
    let mut horizon = 96usize;
    let mut reps = 5usize;
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
            other => return Err(anyhow!("unknown argument: {other}")),
        }
    }

    match cmd.as_str() {
        "forecast" => {
            check_backend_flag(&backend)?;
            cmd_forecast(&fixtures, &weights, config.as_deref(), fixture_idx)
        }
        "parity" => {
            check_backend_flag(&backend)?;
            cmd_parity(&fixtures, &weights, config.as_deref())
        }
        "export-gguf" => cmd_export_gguf(&weights, config.as_deref().ok_or_else(|| anyhow!("--config is required"))?, &quant, &out),
        "bench" => {
            check_backend_flag(&backend)?;
            cmd_bench(&weights, config.as_deref(), n_signals, t_ctx, horizon, reps)
        }
        _ => Err(anyhow!(
            "usage: t0-cli <forecast --fixture N|parity|bench --signals N|export-gguf --quant f16|q8_0|q4_0 --out FILE> \
             [--fixtures DIR] [--weights PATH] [--config PATH] [--backend ndarray|wgpu]"
        )),
    }
}
