//! `t0-cli`: native driver for `t0-core`. Backend is picked by Cargo
//! feature (`ndarray` default, `wgpu` optional) — never inside the library
//! crate, per this repo's CLAUDE.md.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use t0_core::{forecast, T0Config, T0Model, Weights};

#[cfg(feature = "ndarray")]
type Backend = burn_ndarray::NdArray<f32>;
#[cfg(all(feature = "wgpu", not(feature = "ndarray")))]
type Backend = burn_wgpu::Wgpu<f32, i32>;

fn device() -> <Backend as burn::tensor::backend::Backend>::Device {
    Default::default()
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

fn load_model(weights_path: &Path, config_path: &Path) -> Result<T0Model<Backend>> {
    let config: T0Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
    let weights = Weights::load(weights_path)?;
    T0Model::load(&weights, config, &device())
}

fn cmd_forecast(fixtures_dir: &Path, weights: &Path, config: &Path, case_idx: usize) -> Result<()> {
    let manifest: Manifest = serde_json::from_str(&std::fs::read_to_string(fixtures_dir.join("manifest.json"))?)?;
    let case = manifest.cases.get(case_idx).ok_or_else(|| anyhow!("no fixture #{case_idx}"))?;
    let context = read_f32(&fixtures_dir.join(&case.context_file))?;
    let model = load_model(weights, config)?;
    let (out, _) = forecast(&model, &context, case.v, manifest.context_len, manifest.horizon, &device(), false);
    println!("forecast {} ({} values): {:?}...", case.name, out.len(), &out[..out.len().min(10)]);
    Ok(())
}

fn cmd_parity(fixtures_dir: &Path, weights: &Path, config: &Path) -> Result<()> {
    let manifest: Manifest = serde_json::from_str(&std::fs::read_to_string(fixtures_dir.join("manifest.json"))?)?;
    let model = load_model(weights, config)?;
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

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut fixtures = PathBuf::from("fixtures");
    let mut weights = PathBuf::from("../../models/hf/theforecastingcompany/t0-alpha/model.safetensors");
    let mut config = PathBuf::from("../../models/hf/theforecastingcompany/t0-alpha/config.json");
    let mut fixture_idx = 0usize;
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
                config = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--fixture" => {
                fixture_idx = args[i + 1].parse()?;
                i += 2;
            }
            other => return Err(anyhow!("unknown argument: {other}")),
        }
    }

    match cmd.as_str() {
        "forecast" => cmd_forecast(&fixtures, &weights, &config, fixture_idx),
        "parity" => cmd_parity(&fixtures, &weights, &config),
        _ => Err(anyhow!("usage: t0-cli <forecast --fixture N|parity> [--fixtures DIR] [--weights PATH] [--config PATH]")),
    }
}
