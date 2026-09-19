//! Loader shape checks against the real `t0-alpha` checkpoint. Needs the
//! weights downloaded to `~/Code/idle-intelligence/models/hf/…` (see
//! README.md) — `#[ignore]`d with a clear reason when absent, per this
//! repo's testing rule (no silent skips).

use std::path::PathBuf;

use burn_ndarray::NdArray;
use t0_core::{T0Config, T0Model, Weights};

type B = NdArray<f32>;

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME must be set");
    PathBuf::from(home).join("Code/idle-intelligence/models/hf/theforecastingcompany/t0-alpha")
}

#[test]
#[ignore = "requires the t0-alpha checkpoint; run `hf download theforecastingcompany/t0-alpha --local-dir ~/Code/idle-intelligence/models/hf/theforecastingcompany/t0-alpha` first, then `cargo test -- --ignored`"]
fn loader_maps_every_tensor_with_the_right_shape() {
    let dir = model_dir();
    let weights_path = dir.join("model.safetensors");
    assert!(weights_path.exists(), "{} not found — see the #[ignore] reason above", weights_path.display());

    let weights = Weights::load(&weights_path).expect("parse safetensors");
    assert_eq!(weights.names().count(), 303, "t0-alpha has 303 tensors (docs/reports/t0-alpha.md §2)");

    let config: T0Config = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    assert_eq!(config.embed_dim, 512);
    assert_eq!(config.num_layers, 24);
    assert_eq!(config.num_heads, 8);
    assert_eq!(config.patch_size, 32);
    assert_eq!(config.head_dim(), 64);

    assert_eq!(weights.shape("patch_encoder.type_embeddings.weight").unwrap(), vec![3, 512]);
    assert_eq!(weights.shape("transformer.layers.0.attention_block.attention.wQKV.weight").unwrap(), vec![1536, 512]);
    assert_eq!(weights.shape("transformer.layers.0.attention_block.attention.q_norm.scale").unwrap(), vec![64]);
    assert_eq!(weights.shape("transformer.layers.0.mlp.0.weight").unwrap(), vec![4096, 512]);
    assert_eq!(weights.shape("transformer.layers.0.mlp.2.weight").unwrap(), vec![512, 2048]);
    assert_eq!(weights.shape("transformer.out_norm.scale").unwrap(), vec![512]);
    assert_eq!(weights.shape("decoder.residual_layer.weight").unwrap(), vec![160, 512]);

    let device = Default::default();
    let model = T0Model::<B>::load(&weights, config, &device).expect("every tensor the model needs must load");
    assert_eq!(model.config.layer_types().len(), 24);
}
