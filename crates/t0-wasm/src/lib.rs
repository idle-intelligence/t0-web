//! `t0-wasm`: `wasm-bindgen` surface over either the Burn reference path
//! (`t0-core`, `burn_impl.rs`) or the no-Burn `t0-fast` path (`fast.rs`).
//! Backend is a compile-time Cargo feature (`ndarray`/`wgpu` -> Burn,
//! `fast` -> `t0-fast`), same convention as `crates/cli`, never a runtime
//! switch inside the library. Both variants export the identical
//! `initBackend`/`T0Wasm.{load,nQuantiles,forecast,forecastBatch}` JS
//! surface (see `README.md`), so `web/worker.js` can pick a `pkg*` build
//! by feature (preferring `pkg-fast` under WebGPU, falling back to
//! `pkg-wgpu` then `pkg`) without any JS-side branching on which engine it
//! got.

#[cfg(not(feature = "fast"))]
mod burn_impl;
#[cfg(not(feature = "fast"))]
pub use burn_impl::*;

#[cfg(feature = "fast")]
mod fast;
#[cfg(feature = "fast")]
pub use fast::*;

#[cfg(feature = "threads")]
pub use wasm_bindgen_rayon::init_thread_pool;
