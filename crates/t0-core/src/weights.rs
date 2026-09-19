//! `model.safetensors` loader. Every tensor in `theforecastingcompany/t0-alpha`
//! is F32 (see `docs/reports/t0-alpha.md` §2); this loader assumes that and
//! fails loudly otherwise rather than silently misreading bytes.

use anyhow::{anyhow, Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use safetensors::SafeTensors;
use std::collections::HashMap;

pub struct Weights {
    tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
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

    fn raw(&self, name: &str) -> Result<&(Vec<usize>, Vec<f32>)> {
        self.tensors.get(name).ok_or_else(|| anyhow!("missing tensor: {name}"))
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
