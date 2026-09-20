// Tiled shared-memory GEMM, Q4_0-resident `w` (see linear_tiled.wgsl and
// linear_q4.wgsl). Same per-tile dequant-once trick as linear_tiled_q8.wgsl.
const TILE: u32 = 16u;

struct Dims {
    m: u32,
    k: u32,
    n: u32,
    act: u32,
    blocks_per_row: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> qs: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> dims: Dims;

var<workgroup> a_tile: array<array<f32, TILE>, TILE>;
var<workgroup> b_tile: array<array<f32, TILE>, TILE>;

fn dequant_q4(row: u32, col: u32) -> f32 {
    let blk = col / 32u;
    let scale = scales[row * dims.blocks_per_row + blk];
    let within = col % 32u;
    let j = within % 16u;
    let word_idx = row * dims.blocks_per_row * 4u + blk * 4u + j / 4u;
    let word = qs[word_idx];
    let byteval = (word >> ((j % 4u) * 8u)) & 0xFFu;
    let nib = select(byteval & 0xFu, (byteval >> 4u) & 0xFu, within >= 16u);
    return (f32(nib) - 8.0) * scale;
}

@compute @workgroup_size(TILE, TILE)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let m0 = wg.y * TILE;
    let n0 = wg.x * TILE;
    let m = m0 + ty;
    let n = n0 + tx;

    var acc: f32 = 0.0;
    var kb: u32 = 0u;
    loop {
        if (kb >= dims.k) {
            break;
        }
        let ka = kb + tx;
        a_tile[ty][tx] = select(0.0, x[m * dims.k + ka], m < dims.m && ka < dims.k);
        let bn = n0 + ty;
        let bk = kb + tx;
        let b_valid = bn < dims.n && bk < dims.k;
        let safe_bn = select(0u, bn, b_valid);
        let safe_bk = select(0u, bk, b_valid);
        b_tile[ty][tx] = select(0.0, dequant_q4(safe_bn, safe_bk), b_valid);
        workgroupBarrier();

        for (var kk: u32 = 0u; kk < TILE; kk = kk + 1u) {
            acc = acc + a_tile[ty][kk] * b_tile[tx][kk];
        }
        workgroupBarrier();
        kb = kb + TILE;
    }

    if (m < dims.m && n < dims.n) {
        acc = acc + b[n];
        if (dims.act == 1u) {
            acc = max(acc, 0.0);
        }
        out[m * dims.n + n] = acc;
    }
}
