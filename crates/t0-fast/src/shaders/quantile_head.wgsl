// Monotonicity via cumsum-of-softplus: q0 = raw0, qi = q0 + sum_{j<=i} softplus(raw_j).
struct Dims { rows: u32, n_q: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= dims.rows) {
        return;
    }
    let base = r * dims.n_q;
    var running = x[base];
    out[base] = running;
    for (var i: u32 = 1u; i < dims.n_q; i = i + 1u) {
        let raw = x[base + i];
        let sp = log(1.0 + exp(raw));
        running = running + sp;
        out[base + i] = running;
    }
}
