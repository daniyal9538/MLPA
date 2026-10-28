# Architecture

This page follows one call of `MLPA.forward` step by step. Notation: `B` batch size, `M` number of modalities,
`T_m` tokens of modality `m`, `dim_m` its encoder's output dim, `d` = `d_model`, `nb` = `n_bottlenecks`.

## 1. Inputs and states

For every modality `m`, `inputs[m]` holds `x (B, T_m, ...)`, an optional `pad (B, T_m)` and an optional `present (B,)`.
A modality is **present** for a sample if it is flagged present and has at least one non-padding token. Otherwise it
is **missing**. In training mode with `p_any > 0`, modality dropout (`dropout.py`) then marks some present modalities
as **dropped**. The result is `state (B, M)` with values `PRESENT=0`, `MISSING=1`, `DROPPED=2`.

## 2. Target masks

For every present modality a target mask `tgt (B, T_m)` is drawn over the valid tokens (`masking.py`) or taken from
`targets=`. Targets are drawn for dropped modalities as well, because their predictors still predict them.

## 3. Context pass (frozen encoder, no gradient into the encoder)

For the samples where modality `m` is `PRESENT` and has at least one visible (valid, non-target) token:

* `context='crop'`: `x_ctx, pad_ctx, pos_ctx = crop(x, visible)`. The visible tokens are packed to the front and the
  sequence is cut to the longest row `T'`. Then `h = encoder(x_ctx, pad_ctx, pos_ctx)`, `(n, T', dim_m)`.
* `context='replace'`: `h = encoder(mask_input(x, tgt), pad, arange(T_m))`. Only the visible positions are kept
  as MBT tokens; targets and padding become MBT padding.

`h` is projected with `Linear(dim_m, d)` and zeroed at padding, giving `xs_m (B, T', d)` and `pads_m (B, T')`.
Samples that are missing, dropped, or have no visible token get an all-padding row.

## 4. Multimodal bottleneck transformer (MBT)

Every modality has its own stack of `n_layers` `nn.TransformerEncoderLayer(d, n_heads, mlp_dim, dropout, gelu)`.

**Stream input.** Stream `m` processes the samples whose state is not `DROPPED`:

* `PRESENT`: `[CLS_m]` followed by its visible tokens.
* `MISSING`: `[EMPTY_m]` only (all content positions padded).

`CLS_m` and `EMPTY_m` are learned vectors, one per modality.

**Uni-modal layers** `i < fusion_layer`: `h_m ← layer_m,i(h_m)`.

**Fusion layers** `i ≥ fusion_layer`: every stream processes the shared bottleneck tokens `b (B, nb, d)` together with
its own tokens:

```
z_m = layer_m,i( [ b[rows_m] ; h_m ] )        # rows_m: samples stream m processes
h_m ← z_m[:, nb:]
b   ← mean over the streams that processed the sample of z_m[:, :nb]
```

The bottleneck tokens start from a learned `(1, nb, d)` parameter. Because modalities talk only through `b`, the
cross-modal information has to squeeze through `nb` tokens. A `MISSING` modality still contributes its `[EMPTY]`-
conditioned update to the average. A `DROPPED` modality contributes nothing and receives no gradient from that sample.

**Fused summary**: `fused = b.mean(1)`, `(B, d)`. This is the embedding returned by `embed`.

## 5. Predictors

For every modality `m` with at least one target in the batch, on the samples that have targets (present *or*
dropped):

```
f   = Linear(d, dim_m)(LayerNorm(fused))                       # fusion token, (n, 1, dim_m)
q_t = mask_token_m + pos_emb_m[t]          for t = 0..T_m-1    # queries, (n, T_m, dim_m)
out = Predictor_m( LayerNorm([f ; q]) , padding = non-target positions )[:, 1:]
pred = out[tgt]                                                 # (n_targets, dim_m)
```

`Predictor_m` is a pre-norm `nn.TransformerEncoder` with `pred_layers` layers at width `dim_m`. The predictor never sees
the modality's own context tokens. All the information about the sample comes through `fused`. This is what forces the
fused embedding to carry what is needed to predict every modality.

## 6. Targets and loss

```
target_m = encoder_m(x, pad, arange(T_m))[tgt]          # clean input, no gradient
loss_m   = mean over target tokens of || normalize(pred) − normalize(target) ||² / dim_m   × dim_m / min_dim
loss     = Σ_m loss_weight_m · loss_m
```

`F.mse_loss` averages over tokens *and* dims. For unit vectors that equals `(2 − 2·cos)/dim_m`. The factor
`dim_m / min_dim` (`scale_loss_by_dim=True`) gives every modality the same loss per unit of cosine similarity
regardless of its encoder width.

The loss of a modality is a mean over **its** target tokens in the batch, so a modality with long sequences does not
dominate one with few tokens. Use `loss_weight` to rebalance explicitly.

## 7. Gradients and the optimizer

Only MLPA's own parameters are trainable: projections, MBT, CLS/EMPTY/bottleneck tokens, fusion tokens, mask tokens,
position embeddings and predictors. If a modality is dropped or missing for the whole batch, its stream is not run at
all and its parameters keep `grad=None`. With `optimizer.zero_grad(set_to_none=True)`, AdamW then skips them for that
step instead of applying weight decay against a zero gradient.

## 8. Embedding at inference

`embed(inputs)` runs steps 1, 3 and 4 with no targets (every valid token visible) and no dropout. The result is the
fused summary. `return_modalities=True` also mean-pools each stream's output tokens (excluding `[CLS]`/`[EMPTY]`) into a
`(B, d)` per-modality embedding, which is zero where the modality is missing.
