"""Training-time modality dropout.

With probability p_any a sample loses at least one of its available (present) modalities: every available modality m
is then dropped with its own probability p_drop[m], redrawn until at least one is dropped and at least one stays
(after max_draws one random candidate is dropped instead). A modality that is missing in the data is never dropped,
nor is one with p_drop[m] = 0, and a sample with fewer than two available modalities loses none.

So p_any sets how often dropout happens at all, and p_drop sets which modalities tend to be dropped when it does. Note
that p_drop is a probability *given* that the sample is selected (and given the redraw condition), not a marginal
probability: the marginal drop rate of a modality also depends on p_any and on the other modalities.
"""
import torch


def sample_modality_dropout(available: torch.Tensor, p_any: float, p_drop: torch.Tensor,
                            generator: torch.Generator = None, max_draws: int = 100) -> torch.Tensor:
    """available: (B, M) bool, True where the modality is present; p_drop: (M,) float. Returns (B, M) bool, True where
    the modality is dropped. Drawn on the CPU (optionally from `generator`), returned on the device of `available`."""
    avail = available.cpu()
    p = torch.as_tensor(p_drop, dtype=torch.float).cpu()
    B, M = avail.shape
    drop = torch.zeros(B, M, dtype=torch.bool)
    cand = avail & (p > 0)[None]
    todo = (avail.sum(1) >= 2) & cand.any(1) & (torch.rand(B, generator=generator) < p_any)
    for _ in range(max_draws):
        if not todo.any():
            break
        draw = cand & (torch.rand(B, M, generator=generator) < p[None])
        ok = todo & draw.any(1) & (avail & ~draw).any(1)
        drop[ok] = draw[ok]
        todo &= ~ok
    if todo.any():                                    # fallback: one random candidate
        score = torch.rand(B, M, generator=generator).masked_fill(~cand, -1.0)
        one = torch.zeros(B, M, dtype=torch.bool).scatter_(1, score.argmax(1, keepdim=True), True)
        drop[todo] = one[todo]
    return drop.to(available.device)
