# t0-alpha (The Forecasting Company): architecture, weights, license

## Summary

t0-alpha is a 101.6M-parameter (confirmed by direct tensor count, see below; card says "~102M") decoder-style patch transformer for probabilistic multivariate time-series forecasting, released Apache-2.0 by The Forecasting Company (TFC) on Hugging Face (`theforecastingcompany/t0-alpha`, [model card](https://huggingface.co/theforecastingcompany/t0-alpha)). It patches raw series into 32-step tokens, alternates 2 time-attention layers with 1 cross-variate ("group") attention layer (24 layers total, `group_every_n: 3`), and emits 5 native quantiles per output patch. Weights are plain F32 safetensors (411 MB total repo, one `model.safetensors` file), confirmed by inspecting the safetensors header directly. TFC also publishes ONNX exports (INT8, 107 MB; FP16, 208 MB) for `onnxruntime-web`; both cards state FP32 compute. Sizes as of `t0-alpha-onnx-int8` revision `1c18a3e2` (2026-09-13), before its 2026-10-18 re-export; the current alpha INT8 export is 103.6 MB. License is a clean Apache-2.0 grant with NOTICE attributions to Datadog's Toto and Amazon's Chronos-2 — no redistribution restriction found, but re-quantizing/rehosting is still a design decision given the NOTICE attribution obligations (see §2).

## 1. Model architecture

Source: `config.json` fetched directly from the repo ([raw](https://huggingface.co/theforecastingcompany/t0-alpha/raw/main/config.json)), the HF model card ([README](https://huggingface.co/theforecastingcompany/t0-alpha/blob/main/README.md)), the GitHub repo README ([theforecastingcompany/tfc-t0](https://github.com/theforecastingcompany/tfc-t0)), and a Towards Data Science architecture writeup ([Time-Series LLMs, Explained with t0-alpha](https://towardsdatascience.com/time-series-llms-explained-with-t0-alpha/)) — the TDS piece is a third-party explainer, not an official source, flagged as such below.

| Field | Value | Source |
|---|---|---|
| Params | 101,641,541 (counted from safetensors header) | direct tensor inspection |
| Layers | 24 | config.json |
| `group_every_n` | 3 → 16 time-attention layers + 8 group(cross-variate)-attention layers | config.json + card |
| `embed_dim` | 512 | config.json |
| `mlp_hidden_dim` | 2048 (nominal); actual `mlp.0` weight is `[4096,512]` | config.json + tensor shapes (see §2, likely fused SwiGLU gate+up) |
| `num_heads` | 8 (head_dim 64, since `q_norm`/`k_norm` scale is `[64]` and `wQKV` is `[1536,512]` = 3×512) | config.json + tensor shapes |
| `patch_size` | 32 timesteps | config.json |
| `quantile_levels` | [0.1, 0.25, 0.5, 0.75, 0.9] | config.json |
| `scaler_use_arcsinh` | true | config.json |
| `dropout` | 0.1 | config.json |
| Attention type | Multi-head (not GQA) with QK-norm, causal within time-attention | tensor shapes (single `k_norm`/`q_norm` per layer sized to head_dim, not per-KV-group) — **not verified against source code**, inferred from weight shapes |
| Positional encoding | RoPE, applied only inside time-attention blocks ("time-aware rotary embeddings" per card); group-attention layers run over the variate axis and are described as not using RoPE | HF card + TDS piece — **not verified**: no `rope`/`freq` tensor exists in safetensors (RoPE is computed, not learned, so this is expected either way) |
| Normalization | Pre-norm RMSNorm (`norm.scale` vectors present, one per attention block and one per FFN block, no bias) | tensor shapes + card |
| FFN | SwiGLU-style (per card); `mlp.0.weight [4096,512]`, `mlp.2.weight [512,2048]` — the 4096-wide intermediate matches a fused gate+up projection (2×2048) then a SiLU-gated reduction to 2048 before the down-projection | tensor shapes; SwiGLU claim from HF card, shape math is my inference — **not verified** against actual module code |
| Patch encoding | Per-patch input dim 96 = patch_size(32) × 3 (value, time index, validity mask), an MLP with residual (`patch_encoder.projection.{mlp,residual_layer}`) into 512-dim tokens, plus a `type_embeddings` table of 3 rows × 512 (likely: target / historical covariate / future covariate) | tensor shapes; semantic labels for the 3 types are my inference — **not verified** |
| Output head | `decoder.mlp` (512→512→160) + `decoder.residual_layer` (512→160), 160 = 5 quantiles × 32-step patch, plus a fixed non-learned `head.quantile_levels` buffer `[5]` | tensor shapes |
| Context / horizon | Univariate `(B,T)`/`(T,)`, multivariate `(B,V,T)`; future covariates `(B,F,context+horizon)`; up to 1024 steps forecast in one forward pass, longer via autoregressive rollout | HF/GitHub card |
| Normalization scheme | Causal running mean/std with arcsinh transform on targets/history; global stats on future covariates | HF card |
| Training data | Not disclosed in card, repo, or README beyond "no dataset-specific feature engineering, no per-series tuning" (TDS characterization) — **not verified**, exact corpora unknown | TDS piece; card is silent |

t0-beta (256M params, embed_dim 1024, 21 quantile levels 0.01–0.99) is a separate, larger successor already on HF ([theforecastingcompany/t0-beta](https://huggingface.co/theforecastingcompany/t0-beta)).

## 2. Weights, config, license

- **HF repo**: `theforecastingcompany/t0-alpha`, files: `model.safetensors` (407 MB, F32), `config.json` (250 B), `README.md`, `LICENSE`, `NOTICE`, `.gitattributes`. Total 411 MB. ([tree](https://huggingface.co/theforecastingcompany/t0-alpha/tree/main))
- **Dtype**: every one of 303 tensors in the safetensors header is `F32` — confirmed by fetching the file's 8-byte header-length prefix + JSON header via HTTP range requests against the HF CDN redirect (no full download needed). No bf16/fp16 in the canonical repo; those only exist in the separate ONNX exports.
- **Tensor names** (representative, per-layer pattern repeats identically for all 24 layers — the time/group distinction is behavioral, not structural; no separate weight class exists for group layers):
  - `patch_encoder.projection.mlp.{hidden_layer,output_layer}.{weight,bias}`, `patch_encoder.projection.residual_layer.{weight,bias}`, `patch_encoder.type_embeddings.weight [3,512]`
  - `transformer.layers.{i}.norm.scale [512]`, `.attention_block.norm.scale [512]`, `.attention_block.attention.wQKV.{weight [1536,512],bias [1536]}`, `.attention_block.attention.wO.{weight [512,512],bias [512]}`, `.attention_block.attention.{q_norm,k_norm}.scale [64]`, `.mlp.0.{weight [4096,512],bias [4096]}`, `.mlp.2.{weight [512,2048],bias [512]}`
  - `decoder.mlp.{hidden_layer,output_layer}.{weight,bias}`, `decoder.residual_layer.{weight [160,512],bias [160]}`, `head.quantile_levels [5]`
- **config.json** (full, verbatim): `dropout 0.1`, `embed_dim 512`, `group_every_n 3`, `mlp_hidden_dim 2048`, `num_heads 8`, `num_layers 24`, `patch_size 32`, `quantile_levels [0.1,0.25,0.5,0.75,0.9]`, `scaler_use_arcsinh true`. No tokenizer/scaler files ship separately — scaling is computed at runtime from causal statistics, not from a saved artifact.
- **License**: Apache License 2.0, full standard text ([LICENSE](https://huggingface.co/theforecastingcompany/t0-alpha/raw/main/LICENSE)). `NOTICE` file states: *"This product includes software developed at Datadog … licensed under the Apache License 2.0. Source: github.com/DataDog/toto"* and *"This product includes software derived from Chronos-2 by Amazon.com, Inc. … licensed under the Apache License 2.0. Source: github.com/amazon-science/chronos-forecasting"* ([NOTICE](https://huggingface.co/theforecastingcompany/t0-alpha/raw/main/NOTICE)). **Verdict (a design decision, flagged below)**: Apache-2.0 permits re-quantization and redistribution of derivative (quantized) weights commercially, provided the NOTICE file's attributions are carried forward in any redistribution and any modifications are stated (Apache-2.0 §4). No field-of-use or non-commercial clause found anywhere in LICENSE/NOTICE/README. The separate ONNX model cards (`t0-alpha-onnx-fp16`, `t0-alpha-onnx-int8`) additionally disclaim warranty on quantized accuracy ("recommend the full PyTorch model for production or consequential use") — this is a quality caveat, not a legal restriction.

## 3. ONNX exports and published accuracy

- **ONNX runtime details** ([t0-alpha-onnx-int8 card](https://huggingface.co/theforecastingcompany/t0-alpha-onnx-int8), [t0-alpha-onnx-fp16 card](https://huggingface.co/theforecastingcompany/t0-alpha-onnx-fp16)): opset 20; INT8 card states "per-channel signed INT8 weight quantization with FP32 activations… weights are stored in INT8, while calculations and expanded runtime weights use FP32" (i.e., dequantize-then-matmul-in-FP32, not true int8 GEMM); FP16 card says weights stored FP16, "runtime calculations use FP32". Both note an accompanying `webgpu-options.js` that "keeps mask operations on WASM while folding constant weight expansion for WebGPU." Both validated against `onnxruntime-web` 1.29.0, Chromium 152, Apple Metal 3 (per fp16 card fetch) — **not independently re-verified**, taken from card text.
- **Published accuracy** (from the HF card): GIFT-Eval ([SalesforceAIResearch/gift-eval](https://github.com/SalesforceAIResearch/gift-eval), 97 configs / 55 datasets / 7 domains) CRPS 0.4941, MASE 0.7240 (normalized to seasonal-naive); fev-bench ([arXiv:2509.26468](https://arxiv.org/pdf/2509.26468), 100 tasks) skill score 42.2. The exact eval protocol (per-dataset splits, context/horizon per task, aggregation formula) is defined by the GIFT-Eval and fev-bench repos themselves, not by TFC — **not independently reproduced here**; TFC's card only reports the aggregate scores, not per-dataset breakdowns.

## 4. Reference implementations & comparisons

- **Official code**: [github.com/theforecastingcompany/tfc-t0](https://github.com/theforecastingcompany/tfc-t0) — PyTorch (`tfc-t0` PyPI package), plus an MLX port (`tfc-t0-mlx`, "inference-only, loads same safetensors") for Apple Silicon. No GGUF. ONNX exports (fp16/int8) are official, hosted as separate HF repos, not community conversions.
- **NOTICE-acknowledged lineage**: Datadog's [Toto](https://github.com/DataDog/toto) and Amazon's [Chronos-2](https://github.com/amazon-science/chronos-forecasting), both Apache-2.0, are explicitly named as sources of derived code; NXAI's TiRex is credited in the HF card text for "architectural innovations" (patch masking) but not in NOTICE, so likely inspiration rather than derived code.
- **Comparisons** (from HF card + TDS piece, not independently benchmarked here):
  - **Chronos-2**: T5-style encoder-decoder, discretizes values into token bins; t0-alpha is decoder-only and regresses continuous quantiles directly — no shared tokenization scheme despite the NOTICE credit.
  - **TimesFM**: also patch-based decoder-only, closest architectural sibling; TFC's card doesn't detail exact divergences beyond quantile head vs TimesFM's own head.
  - **Moirai**: masked-encoder (not causal decoder), handles arbitrary variate counts via a different mechanism than t0-alpha's fixed group-attention-every-3-layers scheme.
  - **Toto**: direct code lineage per NOTICE; likely shares patch/scaler/attention primitives most closely of any peer, though exact overlap wasn't diffed here.
  - **TiRex**: much smaller (35M vs 101.6M), reported slightly ahead on aggregate CRPS per TDS piece — gap "within run-to-run variation."

## References

- https://huggingface.co/theforecastingcompany/t0-alpha (model card, config.json, LICENSE, NOTICE, safetensors)
- https://huggingface.co/theforecastingcompany/t0-beta
- https://huggingface.co/theforecastingcompany/t0-alpha-onnx-fp16
- https://huggingface.co/theforecastingcompany/t0-alpha-onnx-int8
- https://github.com/theforecastingcompany/tfc-t0
- https://github.com/theforecastingcompany/tfc-t0/blob/main/notebooks/t0-alpha.ipynb
- https://pypi.org/project/tfc-t0/
- https://pypi.org/project/tfc-t0-mlx/
- https://towardsdatascience.com/time-series-llms-explained-with-t0-alpha/ (third-party explainer, not official)
- https://github.com/SalesforceAIResearch/gift-eval
- https://arxiv.org/pdf/2509.26468 (fev-bench)
- https://github.com/DataDog/toto
- https://github.com/amazon-science/chronos-forecasting
