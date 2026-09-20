# t0-alpha anatomy: facts sheet

Sources: `models/hf/theforecastingcompany/{t0-alpha,t0-beta}/{config.json,README.md}`;
`model.safetensors` via `safetensors.safe_open`; ground-truth source
`t0-web/.venv/.../site-packages/t0/` (`tfc-t0==0.5.0`, matches
`github.com/theforecastingcompany/tfc-t0`); port `crates/t0-core/src/{weights,gguf}.rs`;
`docs/BENCHMARKS.md`; `docs/reports/t0-published-numbers.md`. Unlabelled = sourced,
"hypothesis" flagged explicitly.

## 1. Input pipeline
No fixed max context; one forward decodes up to `max_horizon=1024` steps
(`model.py:44`). Patcher left-pads context to a patch boundary, right-pads horizon to
whole patches (`patcher.py`). Scaler: `CausalScaler`, causal Welford mean/std per row,
reset at `group_ids` boundaries; target/historical rows -> causal stats, future
covariates -> per-row **global** (non-causal) stats (`scaler.py`). `use_arcsinh=True`
both models: `arcsinh((x-loc)/scale)`. `scaler_eps`: alpha absent from config.json ->
default `0.1`, `variance_offset` (`sqrt(var+eps)`); beta explicit `0.01`, `std_clamp`
(`sqrt(var).clamp(min=eps)`) — modes diverge below `eps`, mismatched runtime/checkpoint
silently degrades forecasts (both READMEs). Patch size 32 (both). Patch -> token:
`[values(32) ‖ time_index(32) ‖ validity(32)]` (96 floats) through a `ResidualBlock`,
plus a learned type embedding (`nn.Embedding(3, embed_dim)`: target/historical/future);
`time_index = arange(32)/32`, constant. Missing/pad: NaN -> `MISSING`; batch-width
padding -> `PAD`; right-side forecast region on target rows -> `WITHHELD`; all-`PAD`
patches excluded from attention. Token-embedding matmul:
`patch_encoder.projection.mlp.hidden_layer.weight` `[512, 96]` — in=96 (3xpatch_size),
out=embed_dim (alpha).

## 2. The block
24 layers, embed_dim 512 (alpha)/1024 (beta), 8 heads, head_dim 64/128, mlp_hidden 2048
(both), `group_every_n=3` -> layer pattern **TIME, TIME, GROUP** repeated (16 time + 8
group of 24; `Transformer._get_layer_types`). QK-norm: per-head `RMSNorm(head_dim)` on
Q,K only (not V), before RoPE, weight-only, eps 1e-8. RoPE variant: **xPos**
(`TimeAwareRotaryEmbedding(use_xpos=True)`, `rotary_embedding_torch`) — RoPE + power-law
scale term, not plain RoPE; position = per-patch sequence index (`seq_dim=-2`), separate
from the patch-encoder's within-patch time index. Time attention (`TimeSelfAttention`):
causal, each variate row attends its own patch history, uses xPos RoPE. Group attention
(`VariateSelfAttention`): rearranges `"v s d -> s v d"` so variates (not time) are the
attention axis at a fixed patch position, no RoPE. Not weight-shared — each layer owns
distinct `wQKV`/`wO`/`q_norm`/`k_norm`. MLP: SwiGLU, `Linear(embed,2*hidden) -> SwiGLU
-> Linear(hidden,embed)`; `SwiGLU`: `gate,x=chunk(2); silu(gate)*x` (gate-first, matches
xFormers ordering, noted in-source). Norms: RMSNorm everywhere (pre-norm on attn block
and MLP, plus final `out_norm`), weight-only, no bias. Residual:
`x = attn_block(x)` (norm+residual internal), then `x = x + mlp(norm(x))` — two
residual adds per layer.
### Per-layer weights (alpha, `[out,in]`)
| Tensor | Shape | Params |
| --- | --- | --- |
| `attention_block.norm.scale` | [512] | 512 |
| `attention.wQKV.{weight,bias}` | [1536,512]/[1536] | 787,968 |
| `attention.q_norm.scale`, `k_norm.scale` | [64]x2 | 128 |
| `attention.wO.{weight,bias}` | [512,512]/[512] | 262,656 |
| `norm.scale` (MLP pre-norm) | [512] | 512 |
| `mlp.0.{weight,bias}` | [4096,512]/[4096] | 2,101,248 |
| `mlp.2.{weight,bias}` | [512,2048]/[512] | 1,049,088 |

Per-layer total 4,202,112 (same shapes for TIME/GROUP) x24 = 100,850,688. Non-layer:
patch-encoder `ResidualBlock` 361,984 + `type_embeddings` 1,536 + decoder
`ResidualBlock` 426,816 + `head.quantile_levels` 5 + `out_norm.scale` 512 = 790,853.
**Grand total 101,641,541** — exact match to the safetensors sum and "~102M" card
figure. 303 tensors in `model.safetensors`.
## 3. Output
`QuantileHead`: direct quantiles, monotonicity via cumsum-of-softplus
(`q_0=raw_0`, `q_i=q_0+Σsoftplus(raw_j)`), not distributional (`head.py`). Alpha: 5
levels `[0.1,0.25,0.5,0.75,0.9]`. Beta: 21 levels (`0.05`..`0.95` step `0.05` + `0.01`,
`0.99` tails). Decoder: `mlp.hidden_layer` [512,512] (embed->embed), `mlp.output_layer`
[160,512] (embed -> patch_size*n_quantiles=32x5=160), `residual_layer` [160,512] (skip,
same io). Reshaped `(patch_size, n_quantiles)` then `QuantileHead`. Horizon/forward: up
to 1024 steps, patch-aligned. Beyond that, `RolloutManager` (`rollout.py`) decodes
`min(round_up(remaining,32), 1024)` more steps per step, appending predictions as
`VALID`; for continuation it expands `n_paths = len(query_quantile_levels)` trajectories
(one per requested quantile) and reduces via `QuantileRolloutReducer`'s per-path
empirical CDF (Chronos-2-derived).
## 4. Quantization map
Published beta INT8 card: per-channel signed INT8 for **96 transformer projection
matrices**; **6 I/O projections** stay FP32. Our reconstruction (`weights.rs`,
`is_quantizable`/`quantize_int8_theirs`): 96 = 4 big per-layer matmuls (`wQKV`, `wO`,
`mlp.0`, `mlp.2`) x24 layers — same set used for our Q8_0/Q4_0 GGUF export. 6 F32
survivors = patch-encoder `ResidualBlock` (`projection.mlp.hidden_layer`,
`.output_layer`, `.residual_layer`) + decoder `ResidualBlock` (same 3 names). "Channel"
= output-feature **rows** (`shape[0]`); `int8_per_channel_roundtrip` computes one scale
per row over all `cols` inputs (`gguf.rs:81-95`):

| Matmul | alpha cols/rows | beta cols/rows |
| --- | --- | --- |
| wQKV | 512 / 1,536 | 1024 / 3,072 |
| wO | 512 / 512 | 1024 / 1,024 |
| mlp.0 | 512 / 4,096 | 1024 / 4,096 |
| mlp.2 | 2,048 / 512 | 2,048 / 1,024 |

Scales/layer: alpha 6,656, beta 9,216; x24 layers = **alpha 159,744, beta 221,184**
scales over 100,663,296 quantized weights (alpha). Our Q8_0 (32-elem blocks, one f16
scale/block, `gguf.rs:97-111`) needs 100,663,296/32 = **3,145,728 scales** — ~20x finer.

## 5. Training (published material only)
| Item | Status |
| --- | --- |
| Data corpora | Not stated anywhere public. |
| Objective | Hypothesis: pinball/quantile loss over the trained grid, inferred only from head design; no loss statement or training script found (`tfc-t0` ships inference only). |
| Trained quantile levels | Stated: alpha 5, beta 21 (config.json, both READMEs). |
| Context/horizon sampling, compute | Not stated / "not currently reported" (both cards). |
| Lineage | Datadog Toto (attn scaffold, xPos, RMSNorm, SwiGLU, Welford scaler), Amazon Chronos-2 (time/variate factorization, triple projection, rollout reduction), NX-AI TiRex (patch masking) — README + copyright headers. |

## 6. Alpha vs beta
| Field | alpha | beta |
| --- | --- | --- |
| Params | 101,641,541 | ~256M |
| embed_dim / head_dim | 512 / 64 | 1024 / 128 |
| Quantile levels | 5 | 21 |
| scaler_eps / mode | 0.1 (default) / `variance_offset` | 0.01 / `std_clamp` |
| GIFT-Eval CRPS / MASE | 0.4941 / 0.7240 | 0.4738 / 0.6865 |
| fev-bench skill | 42.2 | 46.65 (README) vs 46.37 (prior card scrape, unreconciled) |
| MLX artifact / runtime pin | 407 MB / any | 1.02 GB / `tfc-t0>=0.5.0` |
| layers/heads/patch/group_every_n/mlp_hidden | 24/8/32/3/2048 (identical both) | — |

## Open questions
- Beta's `model.safetensors` not enumerated this pass — shapes come from `config.json`/
  `T0Config.large()`, not confirmed against the file.
- `t0/mask.py` (`build_time_mask`/`build_group_mask`) not read in full — exact
  cross-group/type masking rule inferred, not verified line by line.
- No public training-data/loss documentation found; §5 objective is a hypothesis.
- Beta fev-bench skill: 46.65 (README) vs 46.37 (prior scrape) — unreconciled.
