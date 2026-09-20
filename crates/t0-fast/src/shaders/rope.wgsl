// In-place interleaved-pair xpos RoPE on a slice (q or k) of the fused QKV
// buffer, matching t0-core's `ops.rs::apply_rope` exactly: cos/sin/scale
// tables are precomputed host-side per index d (already carrying the
// cos/sin interleaved-duplication and the scale halves-concat-duplication
// conventions), so this kernel just indexes them directly by d.
// invert_scale != 0 means the table's scale was already inverted host-side
// (k's `scale**-1`) -- this kernel doesn't need to know which.
struct Dims { rows: u32, heads: u32, head_dim: u32, base_offset: u32, row_stride: u32, seq_len: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read_write> buf: array<f32>;
@group(0) @binding(1) var<storage, read> cos_t: array<f32>;
@group(0) @binding(2) var<storage, read> sin_t: array<f32>;
@group(0) @binding(3) var<storage, read> scale_t: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half = dims.head_dim / 2u;
    let total = dims.rows * dims.heads * half;
    let id = gid.x;
    if (id >= total) {
        return;
    }
    let j = id % half;
    let vh = id / half;
    let head = vh % dims.heads;
    let row = vh / dims.heads;
    let seq_idx = row % dims.seq_len;

    let base = row * dims.row_stride + dims.base_offset + head * dims.head_dim;
    let d0 = 2u * j;
    let d1 = d0 + 1u;
    let x0 = buf[base + d0];
    let x1 = buf[base + d1];
    let tbl_base = seq_idx * dims.head_dim;
    let c0 = cos_t[tbl_base + d0];
    let c1 = cos_t[tbl_base + d1];
    let s0 = sin_t[tbl_base + d0];
    let s1 = sin_t[tbl_base + d1];
    let sc0 = scale_t[tbl_base + d0];
    let sc1 = scale_t[tbl_base + d1];

    // rotate_half_interleaved: pair (x0,x1) -> (-x1, x0)
    buf[base + d0] = x0 * c0 * sc0 + (-x1) * s0 * sc0;
    buf[base + d1] = x1 * c1 * sc1 + x0 * s1 * sc1;
}
