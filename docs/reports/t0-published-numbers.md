# The Forecasting Company: every published number, cited

Read-only research, 2026-09-19. Sources: the Hugging Face model cards and
the `theforecastingcompany` GitHub organization. Every number below has
its source.

## t0-alpha — `theforecastingcompany/t0-alpha` (HF, accessed 2026-09-19)

- Params: "approximately 102M". Architecture: 24 layers (16 time-attention,
  8 group-attention), embed_dim 512, feedforward dim 2048, 8 heads, patch
  size 32, dropout 0.1.
- Native quantile levels: 0.1, 0.25, 0.5, 0.75, 0.9.
- GIFT-Eval: CRPS 0.4941, MASE 0.7240.
- fev-bench: skill score 42.2 (vs Seasonal Naive baseline).
- "Horizons up to 1024 timesteps are decoded in one forward pass"; longer
  horizons use autoregressive rollout (no stated upper limit).
- MLX quantized artifact: 407 MB.
- License: Apache-2.0.
- No latency/timing numbers on this card.

## t0-beta — `theforecastingcompany/t0-beta` (HF, accessed 2026-09-19)

- Params: "approximately 256M". Architecture: 24 layers (16 time, 8 group),
  embed_dim 1024, feedforward dim 2048, 8 heads, patch size 32.
- Native quantile levels: 21, from 0.01 to 0.99.
- GIFT-Eval: CRPS 0.4738, MASE 0.6865 ("normalized geometric means over all
  97 configurations").
- fev-bench: skill score 46.37.
- Same "up to 1024 timesteps in one forward pass" horizon statement as alpha.
- MLX quantized artifact: 1.02 GB.
- License: Apache-2.0. Runtime requirements: `tfc-t0>=0.5.0` (PyTorch
  `>=2.4`, Python `>=3.10`) or `tfc-t0-mlx>=0.1.0`.
- No latency/timing numbers on this card.

## t0-alpha-onnx-fp16 — `theforecastingcompany/t0-alpha-onnx-fp16` (HF)

- Artifact `t0-alpha-grouped-fp16.onnx`: 208.1 MB, "FP16 weights, FP32
  compute" ("reduces download size, not runtime weight memory").
- "Seventeen CPU/WebGPU parity cases covered real and synthetic series";
  quote: "These checks measure numerical consistency, not forecasting
  accuracy."
- Compute horizon must be a multiple of 32; graph does not include
  autoregressive rollout beyond 1024 steps.
- "For production or consequential use, we recommend t0-alpha" (i.e. the
  PyTorch original, not this export). "Provided as-is."

## t0-alpha-onnx-int8 — `theforecastingcompany/t0-alpha-onnx-int8` (HF)

Revision `1c18a3e2` (2026-09-13), the one measured in this repo, before the
2026-10-18 re-export; the current export is 103.6 MB.

- Artifact `t0-alpha-grouped-int8.onnx`: 107.2 MB.
- Quantization: "per-channel signed INT8 weight quantization with FP32
  activations" — no per-matrix breakdown given on this card (contrast with
  the beta INT8 card below, which does give one). The card also states,
  of this export generally: **"INT8-weight, FP32-compute."** Compute
  precision is therefore the same class as our own Q8_0 (INT8-class
  weights, F32 compute) — see `docs/BENCHMARKS.md`'s drift and latency
  sections for what that means for the comparisons there.
- Validation: 17 CPU/WebGPU parity cases; context range tested 1–4096,
  target rows 1–64, covariate rows 0–64, compute horizons 32–1024; ONNX
  opset 20. No drift numbers on this card (drift is only quantified on the
  beta INT8 card).
- Card states this derivative "has not been tested as broadly as the full
  model."

## t0-beta-onnx-fp16 — `theforecastingcompany/t0-beta-onnx-fp16` (HF)

- Artifact: 511.9 MB.
- **Timing quote (verbatim)**: "Recorded timings include first-use
  compilation and are not warm-inference benchmarks." No warm-inference
  timing is published; this statement is on this card, not the INT8 one.
- Browser validation: "Browser outputs matched native CPU outputs at
  `rtol=0.001`, `atol=0.001`" across 12 test cases, on "single-thread WASM
  and WebGPU, with ONNX Runtime Web 1.29.0, Chromium 152 and Apple Metal 3."
- Drift vs PyTorch (54 synthetic cases, their own generator — see below):
  "worst per-series mean drift was 0.0146% and worst point drift was
  0.402%" — i.e. the FP16 export's own drift numbers, distinct from the
  INT8 card's 0.2271%/9.393%. Recorded here
  because it is a published number.

## t0-beta-onnx-int8 — `theforecastingcompany/t0-beta-onnx-int8` (HF)

Card revision `6e43e99c` (2026-09-17), before the 2026-10-18 re-export.

- Artifact `t0-beta-grouped-int8.onnx`: 269.1 MB (SHA-256
  `2c517869aac22518c1e04a35386a739d2656490574a40ea9a31187f4f115ac9b`).
- Quantization recipe (verbatim/paraphrased): "per-channel signed INT8
  weights" for **96 transformer projection matrices**; **six input/output
  projection matrices** remain FP32 "to preserve the extreme quantiles";
  storage/computation "INT8 / FP32". Quote: "Compact weights reduce
  download size; expanded runtime weights and calculations use FP32, so
  this is not a proportional reduction in runtime memory."
- **Drift metric definition (verbatim)**: "absolute error divided by the
  reference series' full forecast range across the horizon and all
  quantiles, not relative error on an individual prediction."
- **Reported drift** (54 synthetic cases): worst per-series mean drift
  **0.2271%**, worst point drift **9.393%**. Acceptance gates: 2% mean,
  10% point drift.
- **No case-generator code or distribution details are published** on this
  card; this is the gap `t0-cli drift`'s own generator fills with a documented
  assumption (`docs/BENCHMARKS.md`).

## GitHub — `github.com/theforecastingcompany/tfc-t0`

- `tfc-t0` (Apache-2.0) is the PyPI package used as this repo's parity
  reference; it carries the same architecture numbers as the HF cards for
  both alpha and beta.
