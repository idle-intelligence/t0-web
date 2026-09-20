// One workgroup per (outer, head) pair; one thread per sequence position
// (seq <= MAX_SEQ). Reads q/k/v straight out of the fused QKV buffer
// ([outer*seq, 3*embed], row-major, head slot at head*head_dim within each
// of the three embed-wide q/k/v segments, head_dim = dims.embed /
// dims.heads), applies the additive mask, softmax, and writes the
// concatenated-heads output ([outer*seq, embed]).
// Model max used to be 32 (max_horizon=1024 / patch_size=32) with 128 as
// headroom for group-attention's seq=variate-chunk-count on a batched
// forecast; long-context rollout (t0_core::forecast_rollout) needs whole
// windows up to context_width+horizon = 8192+1024 = 9216 -> 288 patches, but
// 256 is WebGPU's hard workgroup-invocation cap (one thread per seq
// position), so windows wider than 256 patches must be chunked by the
// caller -- t0-fast doesn't do that itself.
const MAX_SEQ: u32 = 256u;

struct Dims { outer: u32, seq: u32, heads: u32, embed: u32, qkv_stride: u32, mask_outer_stride: u32, scale: f32, _pad0: u32 };

@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let owh = wg.x;
    let outer = owh / dims.heads;
    let head = owh % dims.heads;
    let i = lid.x;
    if (i >= dims.seq) {
        return;
    }
    let head_dim = dims.embed / dims.heads;

    let q_base = (outer * dims.seq + i) * dims.qkv_stride + head * head_dim;
    var scores: array<f32, MAX_SEQ>;
    let mask_base = outer * dims.mask_outer_stride + i * dims.seq;

    for (var j: u32 = 0u; j < dims.seq; j = j + 1u) {
        let k_base = (outer * dims.seq + j) * dims.qkv_stride + dims.embed + head * head_dim;
        var acc: f32 = 0.0;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            acc = acc + qkv[q_base + d] * qkv[k_base + d];
        }
        scores[j] = acc * dims.scale + mask[mask_base + j];
    }

    var maxv: f32 = scores[0];
    for (var j: u32 = 1u; j < dims.seq; j = j + 1u) {
        maxv = max(maxv, scores[j]);
    }
    var sum: f32 = 0.0;
    for (var j: u32 = 0u; j < dims.seq; j = j + 1u) {
        let e = exp(scores[j] - maxv);
        scores[j] = e;
        sum = sum + e;
    }
    for (var j: u32 = 0u; j < dims.seq; j = j + 1u) {
        scores[j] = scores[j] / sum;
    }

    let out_base = (outer * dims.seq + i) * dims.embed + head * head_dim;
    for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
        var acc: f32 = 0.0;
        for (var j: u32 = 0u; j < dims.seq; j = j + 1u) {
            let v_base = (outer * dims.seq + j) * dims.qkv_stride + 2u * dims.embed + head * head_dim;
            acc = acc + scores[j] * qkv[v_base + d];
        }
        out[out_base + d] = acc;
    }
}
