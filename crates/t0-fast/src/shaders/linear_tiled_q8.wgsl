// Tiled shared-memory GEMM, Q8_0-resident `w` (see linear_tiled.wgsl for
// the shared-tile trick and linear_q8.wgsl for the block-decode math): the
// B tile is dequantized once per 16x16 block instead of once per output
// thread, so a batch of M rows shares the dequant work across its whole
// tile column instead of repeating it per row.
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

fn dequant_q8(row: u32, col: u32) -> f32 {
    let blk = col / 32u;
    let scale = scales[row * dims.blocks_per_row + blk];
    let within = col % 32u;
    let word = qs[row * (dims.k / 4u) + blk * 8u + within / 4u];
    let shift = (within % 4u) * 8u;
    let byteval = (word >> shift) & 0xFFu;
    let sv = (i32(byteval) << 24u) >> 24u;
    return f32(sv) * scale;
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
        b_tile[ty][tx] = select(0.0, dequant_q8(safe_bn, safe_bk), b_valid);
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
