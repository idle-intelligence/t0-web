// out[r, :] = x[r, :] / sqrt(mean(x[r,:]^2) + eps) * scale[:], dim = `dim`.
struct Dims { rows: u32, dim: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

const EPS: f32 = 1e-8;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= dims.rows) {
        return;
    }
    let base = r * dims.dim;
    var ss: f32 = 0.0;
    for (var d: u32 = 0u; d < dims.dim; d = d + 1u) {
        let v = x[base + d];
        ss = ss + v * v;
    }
    let denom = sqrt(ss / f32(dims.dim) + EPS);
    for (var d: u32 = 0u; d < dims.dim; d = d + 1u) {
        out[base + d] = (x[base + d] / denom) * scale[d];
    }
}
