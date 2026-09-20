// SwiGLU gate: in is [rows, 2*hidden] (gate = [:hidden], up = [hidden:]),
// out is [rows, hidden] = silu(gate) * up.
struct Dims { rows: u32, hidden: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    let total = dims.rows * dims.hidden;
    if (id >= total) {
        return;
    }
    let row = id / dims.hidden;
    let c = id % dims.hidden;
    let base = row * 2u * dims.hidden;
    let gate = src[base + c];
    let up = src[base + dims.hidden + c];
    let silu = gate / (1.0 + exp(-gate));
    out[id] = silu * up;
}
