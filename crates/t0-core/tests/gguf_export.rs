//! Round-trip safetensors -> GGUF -> dequant-at-load, checked against the
//! reference PyTorch quantile fixtures. f16 is gated (<=1e-3 max-abs, per
//! this milestone's brief); Q8_0/Q4_0 error is recorded (printed), not
//! gated — the accuracy verdict for those belongs to the GIFT-Eval step.
//! `#[ignore]`d: needs the t0-alpha checkpoint (never committed).

use std::path::PathBuf;

use burn_ndarray::NdArray;
use serde::Deserialize;
use t0_core::weights::Quant;
use t0_core::{forecast, T0Config, T0Model, Weights};

type B = NdArray<f32>;

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
    context_file: String,
    quantiles_file: String,
}

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn max_abs_rel(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        let abs = (x - y).abs();
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(abs / y.abs().max(1e-6));
    }
    (max_abs, max_rel)
}

#[test]
#[ignore = "requires the t0-alpha checkpoint; run `hf download theforecastingcompany/t0-alpha --local-dir ~/Code/idle-intelligence/models/hf/theforecastingcompany/t0-alpha` first, then `cargo test -- --ignored`"]
fn gguf_round_trip_quantiles() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
    let manifest: Manifest = serde_json::from_str(&std::fs::read_to_string(root.join("manifest.json")).unwrap()).unwrap();

    let home = std::env::var("HOME").unwrap();
    let model_dir = PathBuf::from(home).join("Code/idle-intelligence/models/hf/theforecastingcompany/t0-alpha");
    assert!(model_dir.join("model.safetensors").exists(), "{} not found — see the #[ignore] reason above", model_dir.display());

    let config: T0Config = serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json")).unwrap()).unwrap();
    let weights = Weights::load(&model_dir.join("model.safetensors")).unwrap();

    let tmp = std::env::temp_dir();
    let device = Default::default();

    for (quant, gate) in [(Quant::F16, Some(1e-3f32)), (Quant::Q8_0, None), (Quant::Q4_0, None)] {
        let out_path = tmp.join(format!("t0-alpha-test-{quant:?}.gguf"));
        weights.export_gguf(&config, quant, &out_path).unwrap();

        let (gguf_weights, gguf_config) = Weights::load_gguf(&out_path).unwrap();
        let model = T0Model::<B>::load(&gguf_weights, gguf_config, &device).unwrap();

        let mut worst_abs = 0.0f32;
        let mut worst_rel = 0.0f32;
        for case in &manifest.cases {
            let context = read_f32(&root.join(&case.context_file));
            let expected = read_f32(&root.join(&case.quantiles_file));
            let (got, _) = forecast(&model, &context, case.v, manifest.context_len, manifest.horizon, &device, false);
            let (abs, rel) = max_abs_rel(&got, &expected);
            println!("{quant:?} {}: max-abs={abs:.6e} max-rel={rel:.6e}", case.name);
            worst_abs = worst_abs.max(abs);
            worst_rel = worst_rel.max(rel);
        }
        println!("{quant:?} worst across fixtures: max-abs={worst_abs:.6e} max-rel={worst_rel:.6e}");
        if let Some(gate) = gate {
            assert!(worst_abs <= gate, "{quant:?} max-abs error {worst_abs:.3e} exceeds gate {gate:.3e}");
        }
        let _ = std::fs::remove_file(&out_path);
    }
}
