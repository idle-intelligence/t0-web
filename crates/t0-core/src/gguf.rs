//! Minimal GGUF v3 container (llama.cpp's format, `ggml.h`/`gguf.py`) plus
//! Q8_0/Q4_0 block quantization matching llama.cpp's own block layout
//! (`ggml_quantize_row_q8_0`/`q4_0` in `ggml-quants.c`): 32 values per block,
//! an f16 scale, then the packed values. This module only needs to read back
//! what it writes (t0-alpha's own checkpoint), so the metadata/tensor-info
//! surface is the subset GGUF actually uses, not every value type in the spec.
//!
//! Tensor dimension order follows ggml's convention: `ne[0]` is the
//! fastest-moving (innermost) axis, so a safetensors `[rows, cols]` row-major
//! tensor is written with `ne = [cols, rows]` (reversed).

use anyhow::{anyhow, bail, Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use half::f16;
use std::collections::HashMap;
use std::io::{Read, Write};

pub const QK: usize = 32;
const ALIGNMENT: u64 = 32;
const MAGIC: &[u8; 4] = b"GGUF";
const VERSION: u32 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q8_0,
}

impl GgmlType {
    fn id(self) -> u32 {
        match self {
            GgmlType::F32 => 0,
            GgmlType::F16 => 1,
            GgmlType::Q4_0 => 2,
            GgmlType::Q8_0 => 8,
        }
    }

    fn from_id(id: u32) -> Result<Self> {
        Ok(match id {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            8 => GgmlType::Q8_0,
            other => bail!("unsupported ggml tensor type id {other}"),
        })
    }

    /// Bytes per block of `QK` elements (or per single element for F32/F16).
    fn block_size_bytes(self) -> usize {
        match self {
            GgmlType::F32 => 4,
            GgmlType::F16 => 2,
            GgmlType::Q4_0 => 2 + QK / 2,
            GgmlType::Q8_0 => 2 + QK,
        }
    }

    fn row_bytes(self, n_elements: usize) -> usize {
        match self {
            GgmlType::F32 => n_elements * 4,
            GgmlType::F16 => n_elements * 2,
            GgmlType::Q4_0 | GgmlType::Q8_0 => {
                assert_eq!(n_elements % QK, 0, "quantized tensors must have a multiple of {QK} elements");
                (n_elements / QK) * self.block_size_bytes()
            }
        }
    }
}

/// Quantize a flat row-major f32 buffer to Q8_0 blocks of `QK` values:
/// `d = amax/127`, `qs[i] = round(x[i]/d)`. Matches llama.cpp's
/// `quantize_row_q8_0_ref`.
pub fn quantize_q8_0(x: &[f32]) -> Vec<u8> {
    assert_eq!(x.len() % QK, 0);
    let mut out = Vec::with_capacity((x.len() / QK) * (2 + QK));
    for block in x.chunks_exact(QK) {
        let amax = block.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        for &v in block {
            let q = (v * id).round().clamp(-128.0, 127.0) as i8;
            out.push(q as u8);
        }
    }
    out
}

pub fn dequantize_q8_0(bytes: &[u8], n_elements: usize) -> Vec<f32> {
    assert_eq!(n_elements % QK, 0);
    let mut out = Vec::with_capacity(n_elements);
    for block in bytes.chunks_exact(2 + QK) {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        for &b in &block[2..2 + QK] {
            out.push(d * (b as i8) as f32);
        }
    }
    out
}

/// Quantize to Q4_0 blocks of `QK` values: `d = max/-8` (signed, symmetric
/// around the value of largest magnitude, matching llama.cpp's sign
/// convention), nibbles pack `j` and `j+16` into one byte (low/high nibble),
/// stored as unsigned `q+8` in `[0,15]`. Matches llama.cpp's
/// `quantize_row_q4_0_ref`.
pub fn quantize_q4_0(x: &[f32]) -> Vec<u8> {
    assert_eq!(x.len() % QK, 0);
    let mut out = Vec::with_capacity((x.len() / QK) * (2 + QK / 2));
    for block in x.chunks_exact(QK) {
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &v in block {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        for j in 0..QK / 2 {
            let x0 = block[j] * id;
            let x1 = block[j + QK / 2] * id;
            let xi0 = ((x0 + 8.5).floor() as i32).clamp(0, 15) as u8;
            let xi1 = ((x1 + 8.5).floor() as i32).clamp(0, 15) as u8;
            out.push(xi0 | (xi1 << 4));
        }
    }
    out
}

pub fn dequantize_q4_0(bytes: &[u8], n_elements: usize) -> Vec<f32> {
    assert_eq!(n_elements % QK, 0);
    let mut out = vec![0.0f32; n_elements];
    for (b, block) in bytes.chunks_exact(2 + QK / 2).enumerate() {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let base = b * QK;
        for j in 0..QK / 2 {
            let byte = block[2 + j];
            let lo = (byte & 0x0f) as f32 - 8.0;
            let hi = (byte >> 4) as f32 - 8.0;
            out[base + j] = d * lo;
            out[base + j + QK / 2] = d * hi;
        }
    }
    out
}

#[derive(Debug, Clone)]
pub enum MetaValue {
    U32(u32),
    F32(f32),
    Bool(bool),
    String(String),
    ArrayF32(Vec<f32>),
}

pub struct TensorPlan {
    pub name: String,
    pub shape: Vec<usize>, // row-major, e.g. [rows, cols]
    pub ty: GgmlType,
    pub data: Vec<u8>, // already-encoded bytes for `ty`
}

pub fn quantize_for(ty: GgmlType, values: &[f32]) -> Vec<u8> {
    match ty {
        GgmlType::F32 => {
            let mut out = Vec::with_capacity(values.len() * 4);
            for &v in values {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out
        }
        GgmlType::F16 => {
            let mut out = Vec::with_capacity(values.len() * 2);
            for &v in values {
                out.extend_from_slice(&f16::from_f32(v).to_le_bytes());
            }
            out
        }
        GgmlType::Q8_0 => quantize_q8_0(values),
        GgmlType::Q4_0 => quantize_q4_0(values),
    }
}

pub fn dequantize_for(ty: GgmlType, bytes: &[u8], n_elements: usize) -> Vec<f32> {
    match ty {
        GgmlType::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        GgmlType::F16 => bytes
            .chunks_exact(2)
            .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        GgmlType::Q8_0 => dequantize_q8_0(bytes, n_elements),
        GgmlType::Q4_0 => dequantize_q4_0(bytes, n_elements),
    }
}

fn write_string(w: &mut impl Write, s: &str) -> Result<()> {
    w.write_u64::<LittleEndian>(s.len() as u64)?;
    w.write_all(s.as_bytes())?;
    Ok(())
}

fn read_string(r: &mut impl Read) -> Result<String> {
    let len = r.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).context("gguf string is not valid utf-8")
}

fn write_meta_value(w: &mut impl Write, v: &MetaValue) -> Result<()> {
    match v {
        MetaValue::U32(x) => {
            w.write_u32::<LittleEndian>(4)?;
            w.write_u32::<LittleEndian>(*x)?;
        }
        MetaValue::F32(x) => {
            w.write_u32::<LittleEndian>(6)?;
            w.write_f32::<LittleEndian>(*x)?;
        }
        MetaValue::Bool(x) => {
            w.write_u32::<LittleEndian>(7)?;
            w.write_u8(if *x { 1 } else { 0 })?;
        }
        MetaValue::String(s) => {
            w.write_u32::<LittleEndian>(8)?;
            write_string(w, s)?;
        }
        MetaValue::ArrayF32(xs) => {
            w.write_u32::<LittleEndian>(9)?; // ARRAY
            w.write_u32::<LittleEndian>(6)?; // element type FLOAT32
            w.write_u64::<LittleEndian>(xs.len() as u64)?;
            for x in xs {
                w.write_f32::<LittleEndian>(*x)?;
            }
        }
    }
    Ok(())
}

fn read_meta_value(r: &mut impl Read) -> Result<MetaValue> {
    let ty = r.read_u32::<LittleEndian>()?;
    Ok(match ty {
        4 => MetaValue::U32(r.read_u32::<LittleEndian>()?),
        6 => MetaValue::F32(r.read_f32::<LittleEndian>()?),
        7 => MetaValue::Bool(r.read_u8()? != 0),
        8 => MetaValue::String(read_string(r)?),
        9 => {
            let elem_ty = r.read_u32::<LittleEndian>()?;
            let len = r.read_u64::<LittleEndian>()? as usize;
            if elem_ty != 6 {
                bail!("gguf: only FLOAT32 arrays are supported by this reader, got type {elem_ty}");
            }
            let mut xs = Vec::with_capacity(len);
            for _ in 0..len {
                xs.push(r.read_f32::<LittleEndian>()?);
            }
            MetaValue::ArrayF32(xs)
        }
        other => bail!("gguf: unsupported metadata value type {other}"),
    })
}

pub struct GgufFile {
    pub metadata: HashMap<String, MetaValue>,
    pub tensors: HashMap<String, (Vec<usize>, GgmlType, Vec<u8>)>,
}

pub fn write_gguf(w: &mut impl Write, metadata: &[(String, MetaValue)], tensors: &[TensorPlan]) -> Result<()> {
    w.write_all(MAGIC)?;
    w.write_u32::<LittleEndian>(VERSION)?;
    w.write_u64::<LittleEndian>(tensors.len() as u64)?;
    w.write_u64::<LittleEndian>(metadata.len() as u64)?;

    for (k, v) in metadata {
        write_string(w, k)?;
        write_meta_value(w, v)?;
    }

    // Compute aligned offsets first (relative to the start of the data blob).
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut cursor: u64 = 0;
    for t in tensors {
        offsets.push(cursor);
        cursor += t.data.len() as u64;
        let rem = cursor % ALIGNMENT;
        if rem != 0 {
            cursor += ALIGNMENT - rem;
        }
    }

    for (t, &offset) in tensors.iter().zip(&offsets) {
        write_string(w, &t.name)?;
        w.write_u32::<LittleEndian>(t.shape.len() as u32)?;
        // ggml order: innermost (last, fastest-varying) dimension first.
        for &dim in t.shape.iter().rev() {
            w.write_u64::<LittleEndian>(dim as u64)?;
        }
        w.write_u32::<LittleEndian>(t.ty.id())?;
        w.write_u64::<LittleEndian>(offset)?;
    }

    // Tensor offsets are relative to the start of the data blob (right after
    // the tensor-info section), so the blob can start writing immediately.
    let mut pos: u64 = 0;
    for (t, &offset) in tensors.iter().zip(&offsets) {
        let pad = offset - pos;
        for _ in 0..pad {
            w.write_u8(0)?;
        }
        w.write_all(&t.data)?;
        pos = offset + t.data.len() as u64;
    }
    Ok(())
}

pub fn read_gguf(r: &mut impl Read) -> Result<GgufFile> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        bail!("not a GGUF file (bad magic)");
    }
    let version = r.read_u32::<LittleEndian>()?;
    if version != VERSION {
        bail!("unsupported GGUF version {version}, expected {VERSION}");
    }
    let tensor_count = r.read_u64::<LittleEndian>()? as usize;
    let meta_count = r.read_u64::<LittleEndian>()? as usize;

    let mut metadata = HashMap::with_capacity(meta_count);
    for _ in 0..meta_count {
        let key = read_string(r)?;
        let value = read_meta_value(r)?;
        metadata.insert(key, value);
    }

    struct Info {
        name: String,
        shape: Vec<usize>,
        ty: GgmlType,
        offset: u64,
    }
    let mut infos = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = read_string(r)?;
        let n_dims = r.read_u32::<LittleEndian>()? as usize;
        let mut ne = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            ne.push(r.read_u64::<LittleEndian>()? as usize);
        }
        ne.reverse(); // back to row-major [rows, cols, ...]
        let ty = GgmlType::from_id(r.read_u32::<LittleEndian>()?)?;
        let offset = r.read_u64::<LittleEndian>()?;
        infos.push(Info { name, shape: ne, ty, offset });
    }

    // Everything from here to EOF is the (offset-addressed) data blob.
    let mut blob = Vec::new();
    r.read_to_end(&mut blob)?;

    let mut tensors = HashMap::with_capacity(infos.len());
    for info in infos {
        let n_elements: usize = info.shape.iter().product();
        let nbytes = info.ty.row_bytes(n_elements);
        let start = info.offset as usize;
        let end = start + nbytes;
        let data = blob
            .get(start..end)
            .ok_or_else(|| anyhow!("gguf: tensor {} out of bounds ({}..{} of {} byte blob)", info.name, start, end, blob.len()))?
            .to_vec();
        tensors.insert(info.name, (info.shape, info.ty, data));
    }

    Ok(GgufFile { metadata, tensors })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * 0.37).sin() * 3.0).collect()
    }

    #[test]
    fn q8_0_round_trip_error_bounded() {
        let x = sample(320);
        let q = quantize_q8_0(&x);
        let y = dequantize_q8_0(&q, x.len());
        let max_abs = x.iter().zip(&y).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        // amax<=3, 127 levels -> step <= 3/127 ~= 0.0236
        assert!(max_abs < 0.03, "q8_0 round-trip error too large: {max_abs}");
    }

    #[test]
    fn q4_0_round_trip_error_bounded() {
        let x = sample(320);
        let q = quantize_q4_0(&x);
        let y = dequantize_q4_0(&q, x.len());
        let max_abs = x.iter().zip(&y).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        // amax<=3, 15 levels -> step <= 3/8 ~= 0.375
        assert!(max_abs < 0.4, "q4_0 round-trip error too large: {max_abs}");
    }

    #[test]
    fn gguf_round_trip_preserves_shape_and_values() {
        let a = sample(64);
        let b = sample(320);
        let tensors = vec![
            TensorPlan {
                name: "norm.scale".to_string(),
                shape: vec![64],
                ty: GgmlType::F32,
                data: quantize_for(GgmlType::F32, &a),
            },
            TensorPlan {
                name: "layer.weight".to_string(),
                shape: vec![10, 32],
                ty: GgmlType::Q8_0,
                data: quantize_for(GgmlType::Q8_0, &b),
            },
        ];
        let metadata = vec![
            ("t0.embed_dim".to_string(), MetaValue::U32(512)),
            ("t0.quantile_levels".to_string(), MetaValue::ArrayF32(vec![0.1, 0.5, 0.9])),
            ("t0.scaler_eps_mode".to_string(), MetaValue::String("variance_offset".to_string())),
        ];
        let mut buf = Vec::new();
        write_gguf(&mut buf, &metadata, &tensors).unwrap();
        let file = read_gguf(&mut buf.as_slice()).unwrap();

        let (shape, ty, data) = &file.tensors["norm.scale"];
        assert_eq!(shape, &vec![64]);
        assert_eq!(*ty, GgmlType::F32);
        let got_a = dequantize_for(*ty, data, 64);
        for (x, y) in a.iter().zip(&got_a) {
            assert!((x - y).abs() < 1e-6);
        }

        let (shape, ty, data) = &file.tensors["layer.weight"];
        assert_eq!(shape, &vec![10, 32]);
        assert_eq!(*ty, GgmlType::Q8_0);
        let got_b = dequantize_for(*ty, data, 320);
        let max_abs = b.iter().zip(&got_b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(max_abs < 0.03);

        match &file.metadata["t0.embed_dim"] {
            MetaValue::U32(512) => {}
            other => panic!("unexpected metadata value: {other:?}"),
        }
    }
}
