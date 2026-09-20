// out[row, e] += table[idx[row], e]. table has `embed` columns.
struct Dims { rows: u32, embed: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> table: array<f32>;
@group(0) @binding(2) var<storage, read> idx: array<u32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = dims.rows * dims.embed;
    if (i >= total) {
        return;
    }
    let row = i / dims.embed;
    let e = i % dims.embed;
    out[i] = out[i] + table[idx[row] * dims.embed + e];
}
