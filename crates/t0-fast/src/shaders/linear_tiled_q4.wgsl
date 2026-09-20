// Tiled shared-memory GEMM, Q4_0-resident `w`, 2x2 register blocking (see
// linear_tiled.wgsl for the blocking scheme and linear_q4.wgsl for the
// block-decode math). Re-measured against the naive kernel after adding
// register blocking (docs/runs/2026-09-20-perf.md's earlier 16x16/1-output
// tiled variant lost to naive for Q4_0; this 2x2-blocked variant amortizes
// each dequantized b_tile element across twice the output columns/rows).
const TM: u32 = 32u;
const TN: u32 = 32u;
const TK: u32 = 16u;
const THREADS: u32 = 256u;

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

var<workgroup> a_tile: array<array<f32, TK>, TM>;
var<workgroup> b_tile: array<array<f32, TK>, TN>;

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

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let m0 = wg.y * TM;
    let n0 = wg.x * TN;

    var acc00: f32 = 0.0;
    var acc01: f32 = 0.0;
    var acc10: f32 = 0.0;
    var acc11: f32 = 0.0;

    var kb: u32 = 0u;
    loop {
        if (kb >= dims.k) {
            break;
        }
        for (var i: u32 = 0u; i < 2u; i = i + 1u) {
            let idx = tid + i * THREADS;
            let mm = idx / TK;
            let kk = idx % TK;
            let m = m0 + mm;
            let ka = kb + kk;
            a_tile[mm][kk] = select(0.0, x[m * dims.k + ka], m < dims.m && ka < dims.k);
        }
        for (var i: u32 = 0u; i < 2u; i = i + 1u) {
            let idx = tid + i * THREADS;
            let nn = idx / TK;
            let kk = idx % TK;
            let n = n0 + nn;
            let bk = kb + kk;
            let valid = n < dims.n && bk < dims.k;
            let safe_n = select(0u, n, valid);
            let safe_bk = select(0u, bk, valid);
            b_tile[nn][kk] = select(0.0, dequant_q4(safe_n, safe_bk), valid);
        }
        workgroupBarrier();

        for (var kk: u32 = 0u; kk < TK; kk = kk + 1u) {
            let a0 = a_tile[ty][kk];
            let a1 = a_tile[ty + 16u][kk];
            let b0 = b_tile[tx][kk];
            let b1 = b_tile[tx + 16u][kk];
            acc00 = acc00 + a0 * b0;
            acc01 = acc01 + a0 * b1;
            acc10 = acc10 + a1 * b0;
            acc11 = acc11 + a1 * b1;
        }
        workgroupBarrier();
        kb = kb + TK;
    }

    let m0_ = m0 + ty;
    let m1_ = m0 + ty + 16u;
    let n0_ = n0 + tx;
    let n1_ = n0 + tx + 16u;
    let relu = dims.act == 1u;

    if (m0_ < dims.m && n0_ < dims.n) {
        var v = acc00 + b[n0_];
        if (relu) { v = max(v, 0.0); }
        out[m0_ * dims.n + n0_] = v;
    }
    if (m0_ < dims.m && n1_ < dims.n) {
        var v = acc01 + b[n1_];
        if (relu) { v = max(v, 0.0); }
        out[m0_ * dims.n + n1_] = v;
    }
    if (m1_ < dims.m && n0_ < dims.n) {
        var v = acc10 + b[n0_];
        if (relu) { v = max(v, 0.0); }
        out[m1_ * dims.n + n0_] = v;
    }
    if (m1_ < dims.m && n1_ < dims.n) {
        var v = acc11 + b[n1_];
        if (relu) { v = max(v, 0.0); }
        out[m1_ * dims.n + n1_] = v;
    }
}
