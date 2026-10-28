# Modality dropout

Modality dropout removes whole modalities from training samples, so the model learns to predict a modality from the
others and produces useful fused embeddings when modalities are missing at test time.

## Parameters

* `MLPA(p_any=...)`: probability that a training sample is **selected** for dropout. `0` (the default) disables
  modality dropout.
* `ModalityConfig(p_drop=...)`: per-modality probability of being dropped **given that the sample is selected**.
  `0` means the modality is never dropped.

## Sampling rule (`mlpa.sample_modality_dropout`)

For each sample independently:

1. Candidates are the modalities that are **present** in the data and have `p_drop > 0`. If the sample has fewer than
   two present modalities, or no candidate, nothing is dropped.
2. With probability `1 − p_any` nothing is dropped.
3. Otherwise each candidate `m` is dropped with probability `p_drop[m]`. The draw is repeated until **at least one**
   modality is dropped and **at least one** present modality remains. After 100 failed draws, one random candidate
   is dropped instead.

So every selected sample loses at least one modality, never all of them, and never one that is already missing.

### Marginal rates

`p_drop` is conditional. The fraction of samples in which a modality is actually dropped also depends on `p_any`, on
the other modalities' `p_drop` (the redraw condition) and on how many modalities each sample has. For example, with
two present modalities and `p_drop = 0.3` each, a selected sample always loses exactly one, so each is dropped in
`p_any / 2` of the samples. Measure the marginal rates on your data with:

```python
drop = sample_modality_dropout(present, p_any, torch.tensor([c.p_drop for c in configs]))
drop.float().sum(0) / present.float().sum(0)
```

## What a dropped modality does

| | `MISSING` (absent in the data) | `DROPPED` (modality dropout) |
|---|---|---|
| MBT stream | runs on the learned `[EMPTY_m]` token | **skipped** for that sample |
| bottleneck average | includes the stream | **excludes** the stream |
| gradient into stream `m` | yes (from `[EMPTY_m]`) | none from that sample |
| predictor `m` | nothing to predict | **predicts its targets** from the other modalities |

The last row is the key point. A dropped modality's targets have to be predicted from the fused summary of the
*other* modalities. That is a cross-modal prediction task, and it is what makes the embedding robust to missing
modalities.

Why skip the stream instead of replacing it with `[EMPTY]`? `[EMPTY]` means "absent in the data", which is information
in itself. Dropout simulates a missing modality *without* telling the model, so the other modalities have to carry
the prediction.

## Train vs. eval

* `model.train()`: dropout is drawn in every `forward` call if `p_any > 0`.
* `model.eval()`: no dropout unless you pass `drop=`.
* `drop=` (a `(B, M)` bool tensor or `{name: (B,) bool}`) forces a specific draw in either mode. Use a fixed draw
  (`sample_modality_dropout` with a seeded generator) to measure a "validation loss with dropout" that is comparable
  across epochs and models.

The training loss with dropout is not comparable to the loss without it, since some targets are predicted without
their own modality. Compare validation losses computed with the same `drop`.

## Choosing values

* Start with `p_any` 0.5–0.7 and `p_drop` ≈ 0.3 for every modality.
* Raise `p_drop` for a **dominant** modality, so the model does not rely on it alone, or for one that is often
  missing at deployment.
* Lower `p_drop`, or set it to 0, for a modality that is always available at deployment or carries too little
  information to be predicted from the others.
* `embed` never applies dropout. Missing modalities at inference are expressed through `present=False`, or by leaving
  the modality out of `inputs`.
