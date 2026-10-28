# MLPA: Missing-modality-aware Latent Prediction Architecture

MLPA learns a **fused embedding of several modalities** on top of **frozen uni-modal encoders**. It is self-supervised
and handles **missing modalities** natively:

* A **multimodal bottleneck transformer (MBT)** fuses the modalities. Each modality has its own transformer stream.
  From a chosen layer on, the streams exchange information only through a few shared **bottleneck tokens**, which are
  pooled into the fused embedding.
* It is trained **JEPA-style** (latent prediction). Part of each modality's tokens are masked. A small per-modality
  **predictor** must predict the frozen encoder's output at the masked positions **from the fused embedding alone**.
  The loss is computed in latent space, never on raw inputs.
* **Missing modalities** are first-class. Any modality can be absent for any sample. It is then represented by a
  learned `[EMPTY]` token, and the sample is still embedded from whatever is present.
* Optional **modality dropout** during training, with a **per-modality drop probability**, teaches the model to
  predict each modality from the others and makes the embedding robust when modalities are missing at test time.

The package is small (one model file plus masking and dropout helpers), depends only on PyTorch, and contains no
encoder code. You bring any encoder, for any modality, as long as it maps tokens to `(B, T, dim)`.

```
 modality 1 ─► frozen encoder ─► visible tokens ─► Linear ─► [CLS₁] stream 1 ─┐         ┌─► predictor 1 ─► ẑ₁ at masked positions
 modality 2 ─► frozen encoder ─► visible tokens ─► Linear ─► [CLS₂] stream 2 ─┼─ MBT ─► fused ─► predictor 2 ─► ẑ₂
 modality 3   (missing)                                      [EMPTY₃]         ─┘ (bottleneck   └─► predictor 3 ─► ẑ₃
                                                                                  tokens)
 loss = Σ_m  w_m · MSE( normalize(ẑ_m), normalize(frozen encoder_m(clean input)[masked positions]) )
```

## Installation

```bash
git clone <this repo> && cd MLPA
pip install -e .                 # only needs torch
pip install -e ".[tutorial,dev]" # + torchvision, scikit-learn, matplotlib, jupyter, pytest
# or, equivalently for the dependencies:
pip install -r requirements.txt && pip install -e .
```

## Quick start

```python
import torch
from mlpa import MLPA, ModalityConfig

# Any frozen encoder with forward(x, pad, pos) -> (B, T, dim); see "Encoder interface" below.
modalities = [
    ModalityConfig('image',   image_encoder,   dim=128, max_len=64, mask_ratio=0.5, p_drop=0.3),
    ModalityConfig('text',    text_encoder,    dim=256, max_len=512, mask_ratio=0.3, mask_strategy='block',
                   context='replace', mask_input=lambda ids, t: ids.masked_fill(t, MASK_ID), p_drop=0.3),
    ModalityConfig('tabular', tabular_encoder, dim=32,  max_len=6,  mask_ratio=0.4, mask_strategy='bernoulli',
                   p_drop=0.1, loss_weight=0.5),
]
model = MLPA(modalities, d_model=128, n_layers=4, n_bottlenecks=4, n_heads=4, p_any=0.7)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)   # only MLPA's own parameters

for batch in loader:
    # {'image': {'x': (B, 64, 768)}, 'text': {'x': (B, T), 'pad': (B, T)}, 'tabular': {'x': ..., 'present': (B,)}}
    out = model(batch)                     # masks + modality dropout are drawn inside
    opt.zero_grad(set_to_none=True)
    out.loss.backward()
    opt.step()

model.eval()
z = model.embed(batch)                     # (B, d_model) fused embedding of the unmasked inputs
```

Runnable examples:

* [`examples/quickstart_synthetic.py`](examples/quickstart_synthetic.py): three synthetic modalities generated
  from one latent, random frozen encoders. Runs in about a minute on a CPU.
* [`notebooks/tutorial_mnist.ipynb`](notebooks/tutorial_mnist.ipynb): MNIST split into three modalities (two patch
  sets and a token-id sequence). It covers pretraining toy encoders, masking strategies (library and custom), cropping
  vs. replacing and why the targets must be hidden, modality dropout, training with and without dropout, linear-probe
  evaluation with missing modalities, and a leak experiment.

## Concepts

### Encoder interface

MLPA never trains the encoders. They are frozen (`requires_grad_(False)`, always `eval()`), kept **outside**
`model.parameters()` and `model.state_dict()`, and moved along with `model.to(...)`. An encoder is any `nn.Module`:

```python
encoder(x, pad, pos) -> (B, T, dim)        # or an object with .last_hidden_state (HuggingFace style)
#   x    (B, T, ...)  token inputs: ids (B, T), feature vectors (B, T, F), ...
#   pad  (B, T) bool  True at padding; attention must ignore these positions
#   pos  (B, T) long  original position of every token; use it for the position embedding
```

`pos` matters because with `context='crop'` the encoder receives only the visible tokens, packed to the front. `pos`
tells it where they came from. Encoders that do not use positions (sets, tabular features with type embeddings, ...)
may ignore it.

### Input format

```python
inputs = {
    name: {
        'x':       tensor (B, T, ...),   # required
        'pad':     bool   (B, T),        # optional, True = padding (default: no padding)
        'present': bool   (B,),          # optional, False = modality missing for that sample
    },                                   #   (default: rows with >= 1 non-padding token)
    ...
}
```

All modalities share the batch dimension `B`. A modality left out of the dict is missing for every sample. Rows
where a modality is missing may hold any values; they are never encoded.

### Modality states

Each modality of each sample is in one of three states (`out.state`, `(B, M)`):

| state | when | MBT stream | bottleneck average | predictor |
|---|---|---|---|---|
| `PRESENT` | data present | `[CLS_m]` + visible tokens | included | predicts the masked positions |
| `MISSING` | absent in the data | only the learned `[EMPTY_m]` token | included | nothing to predict |
| `DROPPED` | modality dropout (training) | skipped entirely, no gradient | excluded | **still predicts** the masked positions, now from the other modalities |

### Masking (targets) and context hiding

For every present modality a **target mask** chooses the tokens to predict (`mask_strategy`, `mask_ratio`). You can
also pass your own masks with `model(inputs, targets={name: (B, T) bool})`. The context pass of the encoder must
**not see the targets**, otherwise their content leaks into the fused embedding and the predictor can copy rather than
predict. `context='crop'` removes them from the input. `context='replace'` lets your `mask_input` overwrite them, e.g.
with `[MASK]` ids. See [docs/masking.md](docs/masking.md).

### Modality dropout

With probability `p_any` a training sample is selected for dropout. Each of its present modalities is then dropped with
its own probability `ModalityConfig.p_drop`, redrawn until at least one modality is dropped and at least one remains.
Dropout is drawn only in `model.train()` mode. `drop=` forces a specific draw, e.g. a fixed draw for validation. See
[docs/modality_dropout.md](docs/modality_dropout.md).

## Parameters

### `MLPA(modalities, ...)`: the fusion model

| parameter | default | what it does |
|---|---|---|
| `modalities` | (required) | list of `ModalityConfig`, one per modality; the order fixes the column order of `out.state` |
| `d_model` | `64` | hidden size of the MBT (all streams and bottleneck tokens), and so the size of the fused embedding. Each modality's encoder output is projected from `dim` to `d_model`. |
| `n_layers` | `4` | number of transformer layers **per modality stream**. Each stream has its own weights, so MBT parameters grow as `n_modalities × n_layers`. |
| `fusion_layer` | `n_layers // 2` | index of the first **fusion** layer. Layers `0 .. fusion_layer-1` are uni-modal: each stream attends only to its own tokens. From `fusion_layer` on, each stream also attends to the bottleneck tokens. `0` fuses from the start; `n_layers-1` fuses only in the last layer. Early fusion means more cross-modal interaction; late fusion means more modality-specific processing first. |
| `n_bottlenecks` | `4` | number of shared **bottleneck tokens**. They are the only channel through which modalities exchange information. Their mean is the fused embedding. Few tokens force the model to compress cross-modal information (the point of bottleneck fusion); more tokens widen the channel. |
| `n_heads` | `4` | attention heads in the MBT layers; must divide `d_model` |
| `mlp_dim` | `128` | hidden size of the feed-forward block in each MBT layer (typically 2–4 × `d_model`) |
| `dropout` | `0.1` | standard dropout inside the MBT layers. This is not modality dropout. |
| `pred_layers` | `2` | transformer layers of each modality's **predictor**. The predictor runs at the encoder's width `dim`. Keep it small: a strong predictor can solve the task without a good fused embedding. |
| `pred_heads` | `2` | attention heads of each predictor (override per modality with `ModalityConfig.pred_heads`); must divide that modality's `dim` |
| `pred_mlp_ratio` | `2.0` | predictor feed-forward size = `pred_mlp_ratio × dim` |
| `pred_dropout` | `0.1` | dropout inside the predictors |
| `p_any` | `0.0` | **modality dropout**: probability that a training sample loses at least one modality. `0` disables modality dropout. |
| `scale_loss_by_dim` | `True` | multiplies each modality's loss by `dim / min_dim`. The normalized MSE of two unit vectors is `(2 − 2·cos)/dim`, so without this a modality with a wider encoder would weigh less at the same cosine similarity. |

### `ModalityConfig(name, encoder, dim, max_len, ...)`: one per modality

| parameter | default | what it does |
|---|---|---|
| `name` | (required) | key of the modality in `inputs`, `out.losses`, `targets=`, `drop=` |
| `encoder` | (required) | the frozen uni-modal encoder (see [Encoder interface](#encoder-interface)) |
| `dim` | (required) | output dim of the encoder; the predictor of this modality works at this width |
| `max_len` | (required) | maximum number of tokens `T` of this modality (size of the predictor's position embedding) |
| `mask_ratio` | `0.5` | fraction of the valid tokens that are **targets** (`'bernoulli'`: per-token probability). Higher means a harder task that relies more on the other modalities. |
| `mask_strategy` | `'random'` | `'random'`: exactly `round(mask_ratio·n)` tokens. `'bernoulli'`: each token independently, which suits a handful of tabular features. `'block'`: one contiguous span, for sequences where neighbours are redundant. |
| `context` | `'crop'` | how targets are hidden from the context pass. `'crop'` removes them (any encoder). `'replace'` calls `mask_input` (for encoders pretrained with a `[MASK]` token). |
| `mask_input` | `None` | `'replace'` only: `f(x, target) -> x` with the targets hidden, e.g. `lambda ids, t: ids.masked_fill(t, MASK_ID)` |
| `p_drop` | `0.3` | **per-modality dropout probability**, given that a sample was selected (`p_any`). `0` = never dropped. Use a higher value for a dominant modality the model would otherwise over-rely on, and a lower one for a modality that is rarely missing in practice. |
| `loss_weight` | `1.0` | weight of this modality's loss in the total |
| `pred_heads` | `None` | per-modality override of `MLPA.pred_heads` |

### `model(inputs, targets=None, drop=None, generator=None, return_predictions=False)`

| argument | what it does |
|---|---|
| `inputs` | see [Input format](#input-format) |
| `targets` | `{name: (B, T) bool}` to use your own target masks; other modalities get masks from their `mask_strategy` |
| `drop` | `None`: sample modality dropout if training and `p_any > 0`. A `(B, M)` bool tensor or `{name: (B,) bool}` forces which modalities are dropped. |
| `generator` | CPU `torch.Generator` for the masks and the dropout draw, e.g. a fixed seed for validation |
| `return_predictions` | also return `{name: (pred, target)}` at the target positions |

It returns an `MLPAOutput` with `loss` (the weighted total), `losses` (per modality, the mean over its target
positions in the batch), `fused` (`(B, d_model)`), `state` (`(B, M)`), `targets` (the masks used) and optionally
`predictions`.

### `model.embed(inputs, return_modalities=False)`

The fused embedding `(B, d_model)` of the **unmasked** inputs, without dropout. Call `model.eval()` first. With
`return_modalities=True` it also returns `{name: (B, d_model)}`, each modality's mean-pooled MBT stream output (zero
where the modality is missing).

### Choosing sizes

Rough starting points: `d_model` 64–256; `n_layers` 4 with `fusion_layer = n_layers // 2`; `n_bottlenecks` 4–8;
`pred_layers` 2. For masking, `mask_ratio` 0.3–0.6 for sequences and `'bernoulli'` ≈ 0.4 for a few tabular tokens. For
modality dropout, `p_any` 0.5–0.7 with `p_drop` ≈ 0.3. Trainable parameters are roughly
`M · n_layers · (4·d_model² + 2·d_model·mlp_dim)` for the MBT, plus `pred_layers · (4 + 2·pred_mlp_ratio) · dim_m²` per predictor.

## Saving and loading

`model.state_dict()` contains only MLPA's own parameters; the frozen encoders are not included. To load, rebuild the
same `ModalityConfig`s (with the same encoders) and `MLPA(...)` arguments, then call `load_state_dict`.

```python
torch.save(model.state_dict(), 'mlpa.pt')
model = MLPA(modalities, **same_kwargs); model.load_state_dict(torch.load('mlpa.pt'))
```

## Repository layout

```
mlpa/
  model.py      MLPA, MBT, ModalityConfig, MLPAOutput, normalized_mse
  masking.py    random / bernoulli / block target masks, crop()
  dropout.py    sample_modality_dropout()
docs/
  architecture.md       the forward pass step by step, with shapes
  masking.md            target masks, crop vs. replace, why targets must be hidden, custom masks
  modality_dropout.md   semantics, marginal rates, gradients, validation
examples/quickstart_synthetic.py
notebooks/tutorial_mnist.ipynb
tests/test_mlpa.py      pytest
requirements.txt        torch + tutorial/test dependencies
pyproject.toml          package metadata (pip install -e .)
```

Run the tests with `pytest`.

## Background

MLPA combines bottleneck fusion from *Attention Bottlenecks for Multimodal Fusion* (Nagrani et al., NeurIPS 2021)
with latent-space masked prediction from I-JEPA (Assran et al., CVPR 2023), extended to any number of modalities,
per-sample missing modalities, and modality dropout.
