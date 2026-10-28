# Masking: targets and context hiding

MLPA's training signal comes from predicting, for each modality, the frozen encoder's output at some **target**
positions, using a fused summary built from the remaining **context** tokens of all modalities. Two choices control
this: which tokens are targets (the target mask) and how the targets are kept out of the context (the context mode).

## Target masks

A target mask is a `(B, T)` bool tensor, True at the positions to predict. Library masks (`mlpa/masking.py`) only ever
select valid (non-padding) tokens, and they guarantee:

* at least one target in every row with at least one valid token;
* at least one context token in every row with at least two valid tokens.

| `mask_strategy` | selects | use for |
|---|---|---|
| `'random'` | exactly `round(mask_ratio · n_valid)` tokens, uniformly | general token sets, image patches |
| `'bernoulli'` | each valid token independently with probability `mask_ratio` | very short modalities, e.g. 2–10 tabular features, where an exact count would always be the same |
| `'block'` | one contiguous span of `round(mask_ratio · n_valid)` tokens at a uniform start | sequences where neighbouring tokens are redundant (text, time series, event sequences). Without a span, a masked token could often be inferred from its neighbours alone. Assumes right padding. |

### Custom masks

Pass any masks you like for some or all modalities:

```python
out = model(inputs, targets={'image': my_2d_block_masks})   # (B, T) bool; other modalities use their strategy
```

For example, use 2-D blocks on an image patch grid (shown in the tutorial notebook), whole visits of an event sequence,
or a fixed set of features. Custom masks are intersected with the valid tokens. The guarantees above are then up to you.

### Reproducible masks

`model(inputs, generator=torch.Generator().manual_seed(s))` draws all masks (and the dropout draw) from that
generator. Re-create it with the same seed every validation run to get a fixed validation task.

## Hiding the targets from the context

The target at a position is `encoder(clean input)[position]`. If the **context** pass of the encoder could see the
target tokens, their content would reach the visible tokens' outputs (through attention, convolution, recurrence or
normalisation), then the MBT and the fused summary. The predictor could then reconstruct the target from information
that was handed to it rather than infer it from the rest of the sample and the other modalities. Training would show a
low loss for the wrong reason, and the fused embedding would learn to compress inputs instead of modelling
relationships. The tutorial notebook demonstrates this (section 7).

MLPA therefore hides the targets **before** the context encoder runs. `ModalityConfig.context` selects how.

### `context='crop'` (default)

The target tokens are removed from the input:

```
tokens     t0  t1  t2  t3  t4  t5  t6  t7
targets         ×   ×               ×
cropped    t0  t3  t4  t5  t7               pos = [0, 3, 4, 5, 7]
```

`mlpa.crop(x, keep)` moves the kept tokens of each row to the front (keeping their order), cuts the batch to the longest
row and returns the original positions. The encoder is called as `encoder(x_ctx, pad_ctx, pos_ctx)` and must use `pos`
for its position embeddings, so each token keeps its identity.

* No leak is possible, whatever the encoder architecture.
* Less compute, since the encoder runs on `(1 − mask_ratio)` of the tokens.
* The encoder sees shorter sequences than at inference. Transformers with explicit position embeddings handle this
  well. For encoders that need the full-length layout, use `'replace'`.

This is how I-JEPA's context encoder works.

### `context='replace'`

Your function overwrites the targets in place, and the encoder sees a full-length input:

```python
ModalityConfig('text', bert_like_encoder, dim=256, max_len=512,
               context='replace', mask_input=lambda ids, tgt: ids.masked_fill(tgt, MASK_ID))
```

Only the visible positions' outputs are passed to the MBT (targets and padding are MBT padding). Use this mode for
encoders pretrained with masked-language modelling: they know the `[MASK]` token and expect inputs in exactly that
form. `mask_input` must remove all information about the target. Replacing the target with a `[MASK]` id, a zero
vector plus a "missing" indicator, or similar is fine. Changing only an attention mask may not be: check that
your encoder cannot see a target through any path other than attention.

### Target pass

In both modes the target pass runs on the clean, full input (`encoder(x, pad, arange(T))`) without gradient, and only
the outputs at the target positions are used.
