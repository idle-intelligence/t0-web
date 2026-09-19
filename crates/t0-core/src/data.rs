//! CPU-side time series representation, mirroring `t0.data.TimeSeries` /
//! `t0.model.layers.patcher.Patcher` (see `tfc-t0` PyPI package). All the
//! padding/patching/mask bookkeeping happens on plain `Vec`s, matching the
//! reference's pattern of doing this cheaply before the model's tensor math.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskType {
    Valid = 0,
    Pad = 1,
    Missing = 2,
    Censored = 3,
    Withheld = 4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariateType {
    Target = 0,
    Historical = 1,
    Future = 2,
}

/// Row-major `[v, t]` time series, matching `t0.data.TimeSeries`.
#[derive(Debug, Clone)]
pub struct TimeSeries {
    pub v: usize,
    pub t: usize,
    pub variates: Vec<f32>,
    pub mask: Vec<i8>,
    /// -1 is the padding sentinel.
    pub group_ids: Vec<i64>,
    /// -1 is the padding sentinel.
    pub variate_type: Vec<i64>,
}

fn round_up(value: usize, multiple: usize) -> usize {
    value.div_ceil(multiple) * multiple
}

impl TimeSeries {
    fn at(v: usize, t: usize, row: usize, col: usize) -> usize {
        debug_assert!(row < v && col < t);
        row * t + col
    }

    /// Build a target-only input from a raw `[v, t_ctx]` context (row-major),
    /// appending `horizon` `WITHHELD` steps to predict. NaN marks a missing
    /// observation. One group id per variate row when `v > 1` (all sharing
    /// group 0, joint/"group" forecasting) — mirrors
    /// `TimeSeries.from_array`'s 3-D-input path (`group_ids=None` ->
    /// `sample_ids.repeat_interleave(n_variates)`, and since we build one
    /// sample at a time, that's a single shared group id).
    pub fn from_context(context: &[f32], v: usize, t_ctx: usize, horizon: usize) -> Self {
        assert_eq!(context.len(), v * t_ctx);
        let t = t_ctx + horizon;
        let mut variates = vec![0.0f32; v * t];
        let mut mask = vec![MaskType::Valid as i8; v * t];
        let group_ids = vec![0i64; v * t];
        let variate_type = vec![VariateType::Target as i64; v * t];

        for row in 0..v {
            for col in 0..t_ctx {
                let idx = Self::at(v, t, row, col);
                let x = context[row * t_ctx + col];
                if x.is_nan() {
                    variates[idx] = 0.0;
                    mask[idx] = MaskType::Missing as i8;
                } else {
                    variates[idx] = x;
                    mask[idx] = MaskType::Valid as i8;
                }
            }
            for col in t_ctx..t {
                let idx = Self::at(v, t, row, col);
                variates[idx] = 0.0;
                mask[idx] = MaskType::Withheld as i8;
            }
        }
        TimeSeries {
            v,
            t,
            variates,
            mask,
            group_ids,
            variate_type,
        }
    }

    /// Pad so context and forecast region both land on whole patches, per
    /// `Patcher.pad`. `context_end` is the first `WITHHELD` column shared by
    /// all target rows (== `t_ctx` for inputs built by `from_context`).
    pub fn pad(&self, patch_size: usize, context_end: usize) -> Self {
        let horizon = self.t - context_end;
        let pad_left = (patch_size - context_end % patch_size) % patch_size;
        let pad_right = round_up(horizon, patch_size) - horizon;
        if pad_left == 0 && pad_right == 0 {
            return self.clone();
        }
        let new_t = pad_left + self.t + pad_right;
        let v = self.v;
        let mut variates = vec![0.0f32; v * new_t];
        let mut mask = vec![MaskType::Pad as i8; v * new_t];
        let mut group_ids = vec![-1i64; v * new_t];
        let mut variate_type = vec![-1i64; v * new_t];

        for row in 0..v {
            let row_group = self.group_ids[row * self.t + self.t - 1];
            let row_type = self.variate_type[row * self.t + self.t - 1];
            for col in 0..self.t {
                let src = row * self.t + col;
                let dst = row * new_t + pad_left + col;
                variates[dst] = self.variates[src];
                mask[dst] = self.mask[src];
                group_ids[dst] = self.group_ids[src];
                variate_type[dst] = self.variate_type[src];
            }
            let is_target = row_type == VariateType::Target as i64;
            for col in 0..pad_right {
                let dst = row * new_t + pad_left + self.t + col;
                group_ids[dst] = row_group;
                variate_type[dst] = row_type;
                mask[dst] = if is_target {
                    MaskType::Withheld as i8
                } else {
                    MaskType::Pad as i8
                };
            }
        }
        TimeSeries {
            v,
            t: new_t,
            variates,
            mask,
            group_ids,
            variate_type,
        }
    }

    /// Reshape `[v, t]` into non-overlapping `[v, p, patch_size]` patches.
    pub fn n_patches(&self, patch_size: usize) -> usize {
        assert_eq!(self.t % patch_size, 0, "series width must be patch-aligned");
        self.t / patch_size
    }
}
