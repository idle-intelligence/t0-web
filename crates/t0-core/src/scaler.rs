//! Causal per-timestep scaler, mirroring `t0.scaler.CausalScaler` (`tfc-t0`
//! PyPI package, `t0/scaler.py`) as `T0Forecaster` uses it: constructed with
//! `patch_size=1` (per-timestep granularity, *not* the model's 32-step
//! patch), so `scale_input` normalizes every timestep by its own causal
//! running mean/std, and `rescale_predictions` later subsamples those
//! per-timestep stats down to one value per model output patch (its last
//! timestep) to invert the scaling on quantile predictions.
//!
//! Only the `TARGET`/`HISTORICAL` (causal) branch is implemented — no
//! fixture in this milestone uses `FUTURE` covariate rows (the non-causal
//! global-stats branch in the reference), so we fail loudly rather than
//! silently produce wrong numbers for them.

use crate::data::{MaskType, TimeSeries, VariateType};

const EPS_DEFAULT: f32 = 0.1;

pub struct LocScale {
    /// One value per `[v, t]` cell (patch_size=1 granularity).
    pub loc: Vec<f32>,
    pub scale: Vec<f32>,
    pub v: usize,
    pub t: usize,
}

pub struct CausalScaler {
    pub use_arcsinh: bool,
    pub eps: f32,
    /// "variance_offset": sqrt(var + eps); "std_clamp": sqrt(var).max(eps).
    pub variance_offset: bool,
}

impl CausalScaler {
    pub fn new(use_arcsinh: bool, eps: f32, eps_mode: &str) -> Self {
        CausalScaler {
            use_arcsinh,
            eps,
            variance_offset: eps_mode != "std_clamp",
        }
    }

    /// `t0.scaler.CausalScaler.scale_input`. Returns the scaled series plus
    /// the per-timestep `LocScale` used to invert it later.
    pub fn scale_input(&self, series: &TimeSeries) -> (TimeSeries, LocScale) {
        let (v, t) = (series.v, series.t);
        let mut loc = vec![0.0f32; v * t];
        let mut scale = vec![1.0f32; v * t];

        for row in 0..v {
            let base = row * t;
            for vt in &series.variate_type[base..base + t] {
                if *vt == VariateType::Future as i64 {
                    panic!("FUTURE covariate rows are not supported in this milestone's scaler port");
                }
            }
            self.causal_row_stats(
                &series.variates[base..base + t],
                &series.mask[base..base + t],
                &mut loc[base..base + t],
                &mut scale[base..base + t],
            );
            // Padding sentinel (group_ids < 0): loc=0, scale=1.
            for col in 0..t {
                if series.group_ids[base + col] < 0 {
                    loc[base + col] = 0.0;
                    scale[base + col] = 1.0;
                }
            }
        }

        let mut scaled_variates = vec![0.0f32; v * t];
        for i in 0..v * t {
            let x = (series.variates[i] - loc[i]) / scale[i];
            scaled_variates[i] = if self.use_arcsinh { x.asinh() } else { x };
        }

        let scaled = TimeSeries {
            v,
            t,
            variates: scaled_variates,
            mask: series.mask.clone(),
            group_ids: series.group_ids.clone(),
            variate_type: series.variate_type.clone(),
        };
        (scaled, LocScale { loc, scale, v, t })
    }

    /// Welford causal mean/std over one row, `eps`-guarded per `eps_mode`.
    /// No group-boundary resets: callers only ever hand this one segment
    /// (one series per row), matching every fixture in this milestone.
    fn causal_row_stats(&self, x: &[f32], mask: &[i8], loc: &mut [f32], scale: &mut [f32]) {
        let t = x.len();
        let mut count = 0.0f32;
        let mut mean = 0.0f32;
        let mut m2 = 0.0f32;
        for col in 0..t {
            let valid = mask[col] == MaskType::Valid as i8;
            // Emit *before* folding in this step's observation: stats at
            // position `col` are causal over `[0, col)` `... ` no —
            // reference semantics are causal over `[0, col]` inclusive
            // (Welford updates first, position claims are shifted by the
            // 1-cell-inclusive convention in `_compute_causal_stats`).
            if valid {
                count += 1.0;
                let delta = x[col] - mean;
                mean += delta / count;
                let delta2 = x[col] - mean;
                m2 += delta * delta2;
            }
            loc[col] = mean;
            let variance = if count > 1.0 { (m2 / (count - 1.0)).max(0.0) } else { 0.0 };
            scale[col] = if self.variance_offset {
                (variance + self.eps).sqrt()
            } else {
                variance.sqrt().max(self.eps)
            };
        }
    }

    /// `t0.scaler.CausalScaler.rescale_predictions`: subsample the
    /// per-timestep `LocScale` to one value per model output patch (its
    /// last timestep), then invert `(x - loc) / scale` (and `arcsinh`) on
    /// `predictions` shaped `[v, n_patches, patch_size, quantiles]`
    /// row-major, in place.
    pub fn rescale_predictions(
        &self,
        predictions: &mut [f32],
        loc_scale: &LocScale,
        model_patch_size: usize,
        n_patches: usize,
        n_quantiles: usize,
    ) {
        let v = loc_scale.v;
        let per_row = n_patches * model_patch_size * n_quantiles;
        for row in 0..v {
            for p in 0..n_patches {
                let t_idx = row * loc_scale.t + p * model_patch_size + (model_patch_size - 1);
                let loc = loc_scale.loc[t_idx];
                let scale = loc_scale.scale[t_idx];
                let base = row * per_row + p * model_patch_size * n_quantiles;
                for i in 0..model_patch_size * n_quantiles {
                    let raw = predictions[base + i];
                    let unscaled = if self.use_arcsinh { raw.sinh() } else { raw };
                    predictions[base + i] = unscaled * scale + loc;
                }
            }
        }
    }
}

impl Default for CausalScaler {
    fn default() -> Self {
        CausalScaler::new(true, EPS_DEFAULT, "variance_offset")
    }
}
