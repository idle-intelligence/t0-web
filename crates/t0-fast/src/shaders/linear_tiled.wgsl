// Tiled shared-memory GEMM: out[M,N] = x[M,K] @ w[N,K]^T + b[N]. x and w
// share the same [rows, K] row-major layout (PyTorch [out,in] needs no
// transpose), so both tiles load identically -- only the accumulate step
// reads the w tile "transposed" (b_tile[tx][kk] instead of [kk][tx]),
// avoiding an actual transpose. One workgroup computes one TILExTILE
// output block; TILE=16 keeps total invocations (256) well under the
// browser's WebGPU workgroup-size cap.
const TILE: u32 = 16u;

struct Dims {
    m: u32,
    k: u32,
    n: u32,
    act: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

var<workgroup> a_tile: array<array<f32, TILE>, TILE>;
var<workgroup> b_tile: array<array<f32, TILE>, TILE>;

@compute @workgroup_size(TILE, TILE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
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
        b_tile[ty][tx] = select(0.0, w[select(0u, bn * dims.k + bk, b_valid)], b_valid);
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
