//! `model.safetensors` loader. Every tensor in `theforecastingcompany/t0-alpha`
//! is F32 (see `docs/reports/t0-alpha.md` §2); this loader assumes that and
//! fails loudly otherwise rather than silently misreading bytes.
//!
//! Also loads/writes the GGUF quantized form (`crate::gguf`): at load time
//! every tensor is dequantized straight to f32 (Burn 0.20's wgpu backend
//! pinned here only implements `FloatElement` for `f32` — see
//! `docs/BENCHMARKS.md` — so there is no lower-precision compute path to
//! dequantize into yet; f16/Q8_0/Q4_0 only shrink the file and load time).

use anyhow::{anyhow, bail, Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::io::Write;

use crate::config::T0Config;
use crate::gguf::{self, GgmlType, MetaValue, TensorPlan};

pub struct Weights {
    tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

/// Per-tensor-name policy: which of the four big per-layer matmuls
/// (`wQKV`, `wO`, `mlp.0`, `mlp.2`) get quantized to the requested scheme.
/// Everything else (patch encoder, type embeddings, decoder, all norms,
/// quantile levels) stays f16: those tensors are either tiny (norms, the
/// quantile-level constant) or directly gate output precision (patch
/// encoder feeds every downstream layer; the decoder produces the quantile
/// values themselves) — see `docs/reports/t0-alpha.md` §5.
fn is_quantizable(name: &str) -> bool {
    name.contains(".attention.wQKV.weight")
        || name.contains(".attention.wO.weight")
        || name.contains(".mlp.0.weight")
        || name.contains(".mlp.2.weight")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quant {
    F16,
    Q8_0,
    Q4_0,
}

impl Quant {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "f16" => Quant::F16,
            "q8_0" => Quant::Q8_0,
            "q4_0" => Quant::Q4_0,
            other => bail!("unknown quant scheme {other}, expected f16|q8_0|q4_0"),
        })
    }

    fn big_matmul_type(self) -> GgmlType {
        match self {
            Quant::F16 => GgmlType::F16,
            Quant::Q8_0 => GgmlType::Q8_0,
            Quant::Q4_0 => GgmlType::Q4_0,
        }
    }
}

/// Recovers `T0Config` from a GGUF file's `t0.*` metadata keys (written by
/// `export_gguf` below). Shared by `Weights::load_gguf_bytes` and
/// `t0-fast`'s own GGUF loader (which keeps the big matmuls' quantized
/// bytes resident instead of dequantizing everything to f32 first).
pub fn config_from_gguf_metadata(meta: &HashMap<String, MetaValue>) -> Result<T0Config> {
    let get_u32 = |k: &str| -> Result<usize> {
        match meta.get(k) {
            Some(MetaValue::U32(v)) => Ok(*v as usize),
            _ => bail!("gguf: missing or wrong-typed metadata key {k}"),
        }
    };
    let get_f32 = |k: &str, default: f32| -> f32 {
        match meta.get(k) {
            Some(MetaValue::F32(v)) => *v,
            _ => default,
        }
    };
    let get_bool = |k: &str, default: bool| -> bool {
        match meta.get(k) {
            Some(MetaValue::Bool(v)) => *v,
            _ => default,
        }
    };
    let get_string = |k: &str, default: &str| -> String {
        match meta.get(k) {
            Some(MetaValue::String(v)) => v.clone(),
            _ => default.to_string(),
        }
    };
    let quantile_levels = match meta.get("t0.quantile_levels") {
        Some(MetaValue::ArrayF32(v)) => v.clone(),
        _ => bail!("gguf: missing t0.quantile_levels metadata"),
    };

    Ok(T0Config {
        embed_dim: get_u32("t0.embed_dim")?,
        num_layers: get_u32("t0.num_layers")?,
        num_heads: get_u32("t0.num_heads")?,
        mlp_hidden_dim: get_u32("t0.mlp_hidden_dim")?,
        patch_size: get_u32("t0.patch_size")?,
        group_every_n: get_u32("t0.group_every_n")? as i64,
        dropout: get_f32("t0.dropout", 0.0),
        quantile_levels,
        scaler_use_arcsinh: get_bool("t0.scaler_use_arcsinh", true),
        scaler_eps: get_f32("t0.scaler_eps", 0.1),
        scaler_eps_mode: get_string("t0.scaler_eps_mode", "variance_offset"),
    })
}

impl Weights {
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let st = SafeTensors::deserialize(&bytes).context("parsing safetensors header")?;
        let mut tensors = HashMap::new();
        for (name, view) in st.tensors() {
            if view.dtype() != safetensors::Dtype::F32 {
                return Err(anyhow!("tensor {name} is {:?}, expected F32", view.dtype()));
            }
            let data: Vec<f32> = view
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            tensors.insert(name, (view.shape().to_vec(), data));
        }
        Ok(Weights { tensors })
    }

    /// Load a GGUF file written by `export_gguf`, dequantizing every tensor
    /// to f32 in memory. Returns the model config recovered from the
    /// `t0.*` metadata keys alongside the weights.
    pub fn load_gguf(path: &std::path::Path) -> Result<(Self, T0Config)> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        Weights::load_gguf_bytes(&bytes)
    }

    /// Same as `load_gguf` but from an in-memory buffer — the entry point
    /// for `t0-wasm`, which receives the GGUF as a `Uint8Array` from JS and
    /// has no filesystem.
    pub fn load_gguf_bytes(bytes: &[u8]) -> Result<(Self, T0Config)> {
        let file = gguf::read_gguf(&mut &bytes[..]).context("parsing gguf")?;

        let mut tensors = HashMap::with_capacity(file.tensors.len());
        for (name, (shape, ty, data)) in &file.tensors {
            let n: usize = shape.iter().product();
            let values = gguf::dequantize_for(*ty, data, n);
            tensors.insert(name.clone(), (shape.clone(), values));
        }

        let config = config_from_gguf_metadata(&file.metadata)?;
        Ok((Weights { tensors }, config))
    }

    /// Auto-detect safetensors vs GGUF by magic bytes and load accordingly.
    /// safetensors has no magic of its own (an 8-byte little-endian header
    /// length), so this checks for GGUF's `"GGUF"` magic first and falls
    /// back to safetensors (which then needs `config_path`).
    pub fn load_auto(path: &std::path::Path, config_path: Option<&std::path::Path>) -> Result<(Self, T0Config)> {
        let mut magic = [0u8; 4];
        {
            let mut f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
            std::io::Read::read_exact(&mut f, &mut magic).context("reading file header")?;
        }
        if &magic == b"GGUF" {
            Weights::load_gguf(path)
        } else {
            let config_path = config_path.ok_or_else(|| anyhow!("--config is required for a safetensors weights file"))?;
            let config: T0Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
            Ok((Weights::load(path)?, config))
        }
    }

    /// Write this (safetensors-loaded, always-F32) set of weights out as
    /// GGUF, quantizing the big per-layer matmuls per `quant` and keeping
    /// everything else at f16. See `is_quantizable` for the exact list.
    pub fn export_gguf(&self, config: &T0Config, quant: Quant, out: &std::path::Path) -> Result<()> {
        let big_ty = quant.big_matmul_type();
        let mut names: Vec<&String> = self.tensors.keys().collect();
        names.sort();

        let mut plans = Vec::with_capacity(names.len());
        for name in names {
            let (shape, data) = &self.tensors[name];
            let ty = if shape.len() == 2 && is_quantizable(name) {
                big_ty
            } else {
                GgmlType::F16
            };
            plans.push(TensorPlan {
                name: name.clone(),
                shape: shape.clone(),
                ty,
                data: gguf::quantize_for(ty, data),
            });
        }

        let metadata = vec![
            ("t0.embed_dim".to_string(), MetaValue::U32(config.embed_dim as u32)),
            ("t0.num_layers".to_string(), MetaValue::U32(config.num_layers as u32)),
            ("t0.num_heads".to_string(), MetaValue::U32(config.num_heads as u32)),
            ("t0.mlp_hidden_dim".to_string(), MetaValue::U32(config.mlp_hidden_dim as u32)),
            ("t0.patch_size".to_string(), MetaValue::U32(config.patch_size as u32)),
            ("t0.group_every_n".to_string(), MetaValue::U32(config.group_every_n as u32)),
            ("t0.dropout".to_string(), MetaValue::F32(config.dropout)),
            ("t0.quantile_levels".to_string(), MetaValue::ArrayF32(config.quantile_levels.clone())),
            ("t0.scaler_use_arcsinh".to_string(), MetaValue::Bool(config.scaler_use_arcsinh)),
            ("t0.scaler_eps".to_string(), MetaValue::F32(config.scaler_eps)),
            ("t0.scaler_eps_mode".to_string(), MetaValue::String(config.scaler_eps_mode.clone())),
        ];

        let mut buf = Vec::new();
        gguf::write_gguf(&mut buf, &metadata, &plans)?;
        let mut f = std::fs::File::create(out).with_context(|| format!("creating {}", out.display()))?;
        f.write_all(&buf)?;
        Ok(())
    }

    /// Simulate The Forecasting Company's published INT8 recipe
    /// (`theforecastingcompany/t0-beta-onnx-int8`: "96 transformer
    /// projection matrices" per-channel signed INT8, "six input/output
    /// projections" FP32) on top of this (F32) `Weights`. The 96 figure is
    /// exactly `4 big matmuls/layer * 24 layers` — the same tensor set
    /// `is_quantizable` already names for our own Q8_0/Q4_0 ladder — and the
    /// six I/O projections are exactly the patch encoder's and decoder's
    /// three `ResidualBlockWeights` matrices each. Everything else is left
    /// untouched (already F32). See `crate::gguf::int8_per_channel_roundtrip`.
    pub fn quantize_int8_theirs(&self) -> Self {
        let mut tensors = HashMap::with_capacity(self.tensors.len());
        for (name, (shape, data)) in &self.tensors {
            let data = if shape.len() == 2 && is_quantizable(name) {
                gguf::int8_per_channel_roundtrip(data, shape[0], shape[1])
            } else {
                data.clone()
            };
            tensors.insert(name.clone(), (shape.clone(), data));
        }
        Weights { tensors }
    }

    /// Estimated file size (bytes) if this INT8 simulation were written out
    /// like the GGUF files (INT8 + one f32 scale/row for the 96 quantized
    /// matrices, f16 for everything else) — no such file is ever written by
    /// `quantize_int8_theirs`, this is purely for the drift table's "file MB"
    /// column, labeled "estimated" there.
    pub fn estimated_int8_theirs_bytes(&self) -> u64 {
        let mut total = 0u64;
        for (name, (shape, data)) in &self.tensors {
            total += if shape.len() == 2 && is_quantizable(name) {
                (shape[0] * shape[1] + shape[0] * 4) as u64
            } else {
                (data.len() * 2) as u64
            };
        }
        total
    }

    fn raw(&self, name: &str) -> Result<&(Vec<usize>, Vec<f32>)> {
        self.tensors.get(name).ok_or_else(|| anyhow!("missing tensor: {name}"))
    }

    /// Raw shape + f32 data for a tensor, with no Burn `Backend` involved.
    /// For `t0-fast`, which uploads weights straight into `wgpu::Buffer`s
    /// and never builds a Burn tensor.
    pub fn get_raw(&self, name: &str) -> Result<(&[usize], &[f32])> {
        let (shape, data) = self.raw(name)?;
        Ok((shape.as_slice(), data.as_slice()))
    }

    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.raw(name)?.0.clone())
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    pub fn get1<B: Backend>(&self, name: &str, device: &B::Device) -> Result<Tensor<B, 1>> {
        let (shape, data) = self.raw(name)?;
        if shape.len() != 1 {
            return Err(anyhow!("tensor {name} has shape {:?}, expected rank 1", shape));
        }
        Ok(Tensor::from_floats(data.as_slice(), device))
    }

    pub fn get2<B: Backend>(&self, name: &str, device: &B::Device) -> Result<Tensor<B, 2>> {
        let (shape, data) = self.raw(name)?;
        if shape.len() != 2 {
            return Err(anyhow!("tensor {name} has shape {:?}, expected rank 2", shape));
        }
        Ok(Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([shape[0], shape[1]]))
    }
}
