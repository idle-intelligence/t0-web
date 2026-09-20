//! Pure-math quantile utilities mirroring `t0/quantile.py`'s
//! `interpolate_quantiles`, `weighted_quantile` and
//! `get_prob_mass_per_quantile_level` (see that file's docstrings for the
//! Chronos-2 provenance). Operate on flat `f32` rows instead of tensors: the
//! rollout calls these a handful of times per forecast (once per AR step,
//! per horizon step), so there is nothing to batch on a device for.
//!
//! Callers must pass quantile levels already sorted ascending — the
//! reference doesn't sort them for the probability-mass computation either,
//! and our two callers (`T0Config::quantile_levels`, CLI-supplied query
//! levels) both hold that invariant.

/// `interpolate_quantiles`: linear interpolation of `orig_values` (given at
/// `orig_levels`) onto `query` levels, extending flat at the ends when
/// `orig_levels` doesn't already cover `[0, 1]`.
pub fn interpolate_quantiles(query: &[f32], orig_levels: &[f32], orig_values: &[f32]) -> Vec<f32> {
    let mut idx: Vec<usize> = (0..orig_levels.len()).collect();
    idx.sort_by(|&a, &b| orig_levels[a].partial_cmp(&orig_levels[b]).unwrap());
    let mut levels: Vec<f32> = idx.iter().map(|&i| orig_levels[i]).collect();
    let mut values: Vec<f32> = idx.iter().map(|&i| orig_values[i]).collect();
    if *levels.first().unwrap() > 0.0 {
        levels.insert(0, 0.0);
        values.insert(0, values[0]);
    }
    if *levels.last().unwrap() < 1.0 {
        levels.push(1.0);
        values.push(*values.last().unwrap());
    }
    query
        .iter()
        .map(|&q| {
            let upper = levels.partition_point(|&l| l <= q).min(levels.len() - 1);
            let lower = upper.saturating_sub(1);
            let (ll, ul) = (levels[lower], levels[upper]);
            let (lv, uv) = (values[lower], values[upper]);
            let diff = ul - ll;
            let w = if diff == 0.0 { 0.0 } else { (q - ll) / diff };
            lv + w * (uv - lv)
        })
        .collect()
}

/// `weighted_quantile`: empirical-CDF quantiles of `samples` under
/// `weights` (need not sum to 1).
pub fn weighted_quantile(query: &[f32], weights: &[f32], samples: &[f32]) -> Vec<f32> {
    let sum: f32 = weights.iter().sum();
    let mut idx: Vec<usize> = (0..samples.len()).collect();
    idx.sort_by(|&a, &b| samples[a].partial_cmp(&samples[b]).unwrap());
    let sorted_samples: Vec<f32> = idx.iter().map(|&i| samples[i]).collect();
    let mut cum = 0.0f32;
    let cumul_weights: Vec<f32> = idx
        .iter()
        .map(|&i| {
            cum += weights[i] / sum;
            cum.clamp(0.0, 1.0)
        })
        .collect();
    interpolate_quantiles(query, &cumul_weights, &sorted_samples)
}

/// `get_prob_mass_per_quantile_level`: trapezoidal probability mass per
/// quantile level, normalized to sum to 1. `levels` must be sorted
/// ascending and strictly inside `(0, 1)`.
pub fn prob_mass(levels: &[f32]) -> Vec<f32> {
    let n = levels.len();
    let mut boundaries = Vec::with_capacity(n + 2);
    boundaries.push(0.0);
    boundaries.extend_from_slice(levels);
    boundaries.push(1.0);
    let mut mass: Vec<f32> = (0..n).map(|i| (boundaries[i + 2] - boundaries[i]) / 2.0).collect();
    let sum: f32 = mass.iter().sum();
    for m in mass.iter_mut() {
        *m /= sum;
    }
    mass
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolate_identity_at_trained_levels() {
        let levels = [0.1, 0.25, 0.5, 0.75, 0.9];
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        let out = interpolate_quantiles(&levels, &levels, &values);
        for (a, b) in out.iter().zip(values.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn prob_mass_sums_to_one() {
        let m = prob_mass(&[0.1, 0.25, 0.5, 0.75, 0.9]);
        let sum: f32 = m.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
