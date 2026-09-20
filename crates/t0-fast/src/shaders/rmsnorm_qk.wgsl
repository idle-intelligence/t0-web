// In-place per-head RMSNorm on a slice of the fused QKV buffer.
// QKV buffer is [rows, 3*embed] with embed = heads*head_dim; a given
// (row, head) vector of `head_dim` values lives at
// row*row_stride + base_offset + head*head_dim (base_offset = 0 for q,
// embed for k). scale has `head_dim` elements (q_norm or k_norm).
struct Dims { rows: u32, heads: u32, head_dim: u32, base_offset: u32, row_stride: u32, _pad0: u32, _pad1: u32, _pad2: u32 };

@group(0) @binding(0) var<storage, read_write> buf: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

const EPS: f32 = 1e-8;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    let total = dims.rows * dims.heads;
    if (n >= total) {
        return;
    }
    let row = n / dims.heads;
    let head = n % dims.heads;
    let base = row * dims.row_stride + dims.base_offset + head * dims.head_dim;
    var ss: f32 = 0.0;
    for (var d: u32 = 0u; d < dims.head_dim; d = d + 1u) {
        let v = buf[base + d];
        ss = ss + v * v;
    }
    let denom = sqrt(ss / f32(dims.head_dim) + EPS);
    for (var d: u32 = 0u; d < dims.head_dim; d = d + 1u) {
        buf[base + d] = (buf[base + d] / denom) * scale[d];
    }
}
