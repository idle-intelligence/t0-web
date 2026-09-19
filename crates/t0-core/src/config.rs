use serde::Deserialize;

/// Mirrors `t0.config.T0Config` (see the `tfc-t0` PyPI package,
/// `t0/config.py`). Field names match `config.json` on
/// `theforecastingcompany/t0-alpha` verbatim.
#[derive(Debug, Clone, Deserialize)]
pub struct T0Config {
    pub embed_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub mlp_hidden_dim: usize,
    pub patch_size: usize,
    pub group_every_n: i64,
    pub dropout: f32,
    pub quantile_levels: Vec<f32>,
    #[serde(default = "default_true")]
    pub scaler_use_arcsinh: bool,
    #[serde(default = "default_scaler_eps")]
    pub scaler_eps: f32,
    #[serde(default = "default_eps_mode")]
    pub scaler_eps_mode: String,
}

fn default_true() -> bool {
    true
}
fn default_scaler_eps() -> f32 {
    0.1
}
fn default_eps_mode() -> String {
    "variance_offset".to_string()
}

impl T0Config {
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads
    }

    pub fn n_quantiles(&self) -> usize {
        self.quantile_levels.len()
    }

    /// Layer types for the 24-layer stack: `TIME, TIME, GROUP` repeated,
    /// per `t0.model.layers.transformer.Transformer._get_layer_types`.
    pub fn layer_types(&self) -> Vec<LayerType> {
        if self.group_every_n <= 0 {
            return vec![LayerType::Time; self.num_layers];
        }
        let n = self.group_every_n as usize;
        let block: Vec<LayerType> = (0..n - 1)
            .map(|_| LayerType::Time)
            .chain(std::iter::once(LayerType::Group))
            .collect();
        let n_blocks = self.num_layers / n;
        block.into_iter().cycle().take(n_blocks * n).collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    Time,
    Group,
}
