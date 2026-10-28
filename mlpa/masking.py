"""Target masks (which tokens each predictor must predict) and context cropping.

A target mask is a (B, T) bool tensor, True at the positions whose frozen-encoder output is predicted. Only valid
(non-padding) tokens are ever selected. For every row with at least one valid token at least one target is drawn, and
for every row with at least two valid tokens at least one token stays visible as context.

Strategies (ModalityConfig.mask_strategy):
  random     exactly round(mask_ratio * n_valid) tokens, uniformly without replacement (clamped to [1, n_valid - 1])
  bernoulli  every valid token independently with probability mask_ratio (at least one is forced); suits very short
             modalities, e.g. a few tabular features
  block      one contiguous span of round(mask_ratio * n_valid) tokens at a uniform start; assumes the valid tokens
             form a prefix of the row (right padding)

The masks are drawn on the CPU (optionally from a torch.Generator, for reproducible validation masks) and moved to the
device of `valid`.
"""
import torch

STRATEGIES = ('random', 'bernoulli', 'block')


def _counts(n_valid: torch.Tensor, ratio: float) -> torch.Tensor:
    k = torch.round(n_valid.float() * ratio).long()
    k = torch.minimum(k, n_valid - 1)                 # keep >= 1 context token when there are >= 2 tokens
    return torch.where(n_valid > 0, k.clamp(min=1), torch.zeros_like(k))


def random_mask(valid: torch.Tensor, ratio: float, generator: torch.Generator = None) -> torch.Tensor:
    """Exactly round(ratio * n_valid) targets per row, uniformly among the valid tokens."""
    v = valid.cpu()
    k = _counts(v.sum(1), ratio)
    score = torch.rand(v.shape, generator=generator).masked_fill(~v, 2.0)      # padding sorts last
    rank = score.argsort(1).argsort(1)
    return (rank < k[:, None]).to(valid.device)


def bernoulli_mask(valid: torch.Tensor, ratio: float, generator: torch.Generator = None) -> torch.Tensor:
    """Every valid token is a target with probability `ratio`; rows that drew none get one at random. A row with
    two or more valid tokens that drew all of them keeps one at random as context."""
    v = valid.cpu()
    tgt = (torch.rand(v.shape, generator=generator) < ratio) & v
    one = random_mask(v, 0.0, generator)                                       # exactly one valid token per row
    tgt = tgt | (one & ~tgt.any(1, keepdim=True))
    full = (tgt == v).all(1) & (v.sum(1) >= 2)
    keep = random_mask(v, 0.0, generator)
    tgt = torch.where(full[:, None], tgt & ~keep, tgt)
    return tgt.to(valid.device)


def block_mask(valid: torch.Tensor, ratio: float, generator: torch.Generator = None) -> torch.Tensor:
    """One contiguous span of round(ratio * n_valid) targets per row (valid tokens must be a prefix)."""
    v = valid.cpu()
    n = v.sum(1)
    k = _counts(n, ratio)
    start = (torch.rand(len(n), generator=generator) * (n - k + 1).float()).long()
    pos = torch.arange(v.shape[1])[None]
    return ((pos >= start[:, None]) & (pos < (start + k)[:, None]) & v).to(valid.device)


def make_mask(valid: torch.Tensor, strategy: str, ratio: float, generator: torch.Generator = None) -> torch.Tensor:
    if strategy == 'random':
        return random_mask(valid, ratio, generator)
    if strategy == 'bernoulli':
        return bernoulli_mask(valid, ratio, generator)
    if strategy == 'block':
        return block_mask(valid, ratio, generator)
    raise ValueError(f'unknown mask strategy {strategy!r}, expected one of {STRATEGIES}')


def crop(x: torch.Tensor, keep: torch.Tensor):
    """Crop masking: move the kept tokens of every row to the front (in their original order) and cut the sequence to
    the longest row. x: (B, T, ...); keep: (B, T) bool. Returns (x_ctx (B, T', ...), pad_ctx (B, T') True = padding,
    pos_ctx (B, T') original position of each kept token, 0 on padding). Padding entries of x_ctx are zeros."""
    B, T = keep.shape
    order = (~keep).to(torch.int8).argsort(dim=1, stable=True)                 # kept tokens first, order preserved
    n = keep.sum(1)
    T_ctx = int(n.max()) if B else 0
    idx = order[:, :T_ctx]
    pad = torch.arange(T_ctx, device=keep.device)[None] >= n[:, None]
    gather_idx = idx.reshape(B, T_ctx, *([1] * (x.dim() - 2))).expand(B, T_ctx, *x.shape[2:])
    x_ctx = x.gather(1, gather_idx)
    x_ctx = x_ctx.masked_fill(pad.reshape(B, T_ctx, *([1] * (x.dim() - 2))), 0)
    return x_ctx, pad, idx.masked_fill(pad, 0)
