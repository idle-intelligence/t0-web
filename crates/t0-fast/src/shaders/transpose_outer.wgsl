// [a, b, e] -> [b, a, e]: swap the two outer axes, copy the inner `e` run.
struct Dims { a: u32, b: u32, e: u32, _pad0: u32 };

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    let total = dims.a * dims.b;
    if (id >= total) {
        return;
    }
    let row = id / dims.b;
    let col = id % dims.b;
    let src_base = (row * dims.b + col) * dims.e;
    let dst_base = (col * dims.a + row) * dims.e;
    for (var k: u32 = 0u; k < dims.e; k = k + 1u) {
        dst[dst_base + k] = src[src_base + k];
    }
}
