"""MLPA: Missing-modality-aware Latent Prediction Architecture.

A multimodal bottleneck transformer (MBT) fuses the outputs of frozen uni-modal encoders and is trained JEPA-style:
per modality, a predictor reconstructs the frozen encoder's output at masked (target) positions from the fused summary
alone. See docs/architecture.md for the full description.

Per modality m:
  1. A target mask is drawn over the valid tokens (masking.py, or passed by the caller).
  2. Context: the frozen encoder runs on the input with the targets hidden, either cropped out (context='crop') or
     replaced by the caller's mask_input (context='replace'). Only the visible tokens' outputs are kept.
  3. They are projected to d_model and form stream m of the MBT, after a learned first token [CLS_m].
  4. Layers before `fusion_layer` are uni-modal; from it on every stream also attends to the shared bottleneck tokens,
     whose updates are averaged over the streams that processed the sample. The last layer's bottleneck tokens are
     mean-pooled into the fused summary (B, d_model).
  5. A fusion token (LayerNorm + Linear of the summary to dim_m) and, at every target position, mask token + position
     embedding go through the modality's predictor. It never sees the modality's own context tokens, only the summary.
  6. Target = the frozen encoder's output for the unmasked input at the position. Loss = MSE of the L2-normalised
     prediction and target, times dim_m / min_dim (with scale_loss_by_dim) so modalities with wider encoders are not
     down-weighted at the same cosine similarity.
Total loss = sum over modalities of loss_weight x mean loss over that modality's target positions in the batch.

State of a modality in a sample:
  PRESENT  stream m gets [CLS_m] + its visible tokens; its target positions are predicted.
  MISSING  (absent in the data) stream m gets only the learned [EMPTY_m] token and takes part in the bottleneck
           average like a present modality; nothing to predict.
  DROPPED  (modality dropout, training only) stream m is skipped for that sample and left out of the bottleneck
           average, so no parameter of stream m gets gradient from it. Its predictor still predicts the same target
           positions, now from the summary of the other modalities only.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dropout import sample_modality_dropout
from .masking import STRATEGIES, crop, make_mask

PRESENT, MISSING, DROPPED = 0, 1, 2


@dataclass
class ModalityConfig:
    """One modality: its frozen encoder and how it is masked, dropped and weighted.

    encoder        nn.Module called as encoder(x, pad, pos) -> (B, T, dim): x (B, T, ...) the token inputs, pad (B, T)
                   bool True at padding, pos (B, T) long the original position of every token. Frozen by MLPA.
    dim            encoder output dim.
    max_len        maximum number of tokens T of this modality (size of the predictor's position embedding).
    mask_ratio     fraction of the valid tokens that are targets (for 'bernoulli': per-token probability).
    mask_strategy  'random' | 'bernoulli' | 'block' (masking.py).
    context        'crop': targets are removed from the encoder input (any encoder; positions passed via pos).
                   'replace': mask_input(x, target) hides them in place (e.g. [MASK] ids for an MLM-trained encoder);
                   the encoder sees the full-length input and only the visible positions' outputs are kept.
    mask_input     'replace' only: (x, target (B, T) bool) -> x with the targets hidden.
    p_drop         modality dropout probability of this modality, given that a sample is selected for dropout (see
                   MLPA.p_any); 0 = never dropped.
    loss_weight    weight of this modality's loss in the total.
    pred_heads     attention heads of this modality's predictor (default MLPA.pred_heads); must divide dim.
    """
    name: str
    encoder: nn.Module
    dim: int
    max_len: int
    mask_ratio: float = 0.5
    mask_strategy: str = 'random'
    context: str = 'crop'
    mask_input: Optional[Callable] = None
    p_drop: float = 0.3
    loss_weight: float = 1.0
    pred_heads: Optional[int] = None


@dataclass
class MLPAOutput:
    loss: torch.Tensor                                   # weighted total
    losses: Dict[str, torch.Tensor]                      # per modality (only those with targets in the batch)
    fused: torch.Tensor                                  # (B, d_model) fused summary
    state: torch.Tensor                                  # (B, M) PRESENT / MISSING / DROPPED
    targets: Dict[str, torch.Tensor]                     # per modality (B, T) bool target masks used
    predictions: Optional[Dict[str, tuple]] = field(default=None)   # name -> (pred, target), (n_targets, dim) each


class MBT(nn.Module):
    """Multimodal bottleneck transformer for any number of modalities, routed by the per-sample state.
    Layers before `fusion_layer` are uni-modal; from `fusion_layer` on, bottleneck fusion."""

    def __init__(self, n_mod: int, d_model: int = 64, n_layers: int = 4, fusion_layer: int = None,
                 n_bottlenecks: int = 4, n_heads: int = 4, mlp_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.L_f = n_layers // 2 if fusion_layer is None else fusion_layer
        self.nb = n_bottlenecks
        assert 0 <= self.L_f < n_layers, f'fusion_layer must be in [0, n_layers), got {self.L_f}'
        assert n_bottlenecks > 0, 'n_bottlenecks must be > 0'
        self.bottlenecks = nn.Parameter(torch.randn(1, n_bottlenecks, d_model))
        self.streams = nn.ModuleList(
            nn.ModuleList(nn.TransformerEncoderLayer(d_model, n_heads, mlp_dim, dropout, 'gelu', batch_first=True)
                          for _ in range(n_layers)) for _ in range(n_mod))
        self.cls = nn.Parameter(torch.randn(n_mod, d_model))           # first token of a present modality
        self.empty = nn.Parameter(torch.randn(n_mod, d_model))         # the only token of a missing modality

    def forward(self, xs: list, pads: list, state: torch.Tensor, return_streams: bool = False):
        """xs[m]: (B, T_m, d) context tokens; pads[m]: (B, T_m) True where there is no token; state: (B, M).
        Returns the bottleneck tokens (B, n_bottlenecks, d), and with return_streams per modality the rows it
        processed, its output tokens (n, 1 + T_m, d) and their padding mask."""
        B = state.shape[0]
        rows, hs, ps = [], [], []
        for m, (x, p) in enumerate(zip(xs, pads)):
            r = torch.nonzero(state[:, m] != DROPPED).squeeze(1)      # the samples stream m processes
            miss = state[r, m] == MISSING
            first = torch.where(miss[:, None], self.empty[m], self.cls[m])[:, None]
            rows.append(r)
            hs.append(torch.cat([first.to(x.dtype), x[r]], 1))
            ps.append(torch.cat([miss.new_zeros(len(r), 1), p[r] | miss[:, None]], 1))
        b = self.bottlenecks.expand(B, -1, -1)
        for i in range(len(self.streams[0])):
            if i < self.L_f:
                for m, r in enumerate(rows):
                    if len(r):
                        hs[m] = self.streams[m][i](hs[m], src_key_padding_mask=ps[m])
                continue
            acc, cnt = torch.zeros_like(b), b.new_zeros(B)
            for m, r in enumerate(rows):
                if not len(r):
                    continue
                z = self.streams[m][i](torch.cat([b[r], hs[m]], 1),
                                       src_key_padding_mask=torch.cat([ps[m].new_zeros(len(r), self.nb), ps[m]], 1))
                acc = acc.index_add(0, r, z[:, :self.nb].to(acc.dtype))
                cnt = cnt.index_add(0, r, torch.ones(len(r), dtype=cnt.dtype, device=cnt.device))
                hs[m] = z[:, self.nb:]
            b = acc / cnt.clamp(min=1)[:, None, None]
        if return_streams:
            return b, list(zip(rows, hs, ps))
        return b


def normalized_mse(pred, target):
    if not len(pred):
        return pred.sum() * 0.0
    return F.mse_loss(F.normalize(pred.float(), dim=-1), F.normalize(target.float(), dim=-1))


class MLPA(nn.Module):
    """MBT + per-modality predictors over frozen uni-modal encoders.

    The encoders are held outside the module's parameters and state_dict (they are frozen and saved separately);
    .to() / .cuda() / .half() move them along, and they always stay in eval mode.

    inputs (forward / embed): {name: {'x': (B, T, ...), 'pad': (B, T) bool True = padding (default: none),
    'present': (B,) bool (default: rows with at least one non-padding token)}}. A modality absent from `inputs` is
    missing for every sample. Rows where a modality is missing may hold any values; they are never encoded.
    """

    def __init__(self, modalities: Sequence[ModalityConfig], d_model: int = 64, n_layers: int = 4,
                 fusion_layer: int = None, n_bottlenecks: int = 4, n_heads: int = 4, mlp_dim: int = 128,
                 dropout: float = 0.1, pred_layers: int = 2, pred_heads: int = 2, pred_mlp_ratio: float = 2.0,
                 pred_dropout: float = 0.1, p_any: float = 0.0, scale_loss_by_dim: bool = True):
        super().__init__()
        names = [c.name for c in modalities]
        assert len(modalities) >= 1 and len(set(names)) == len(names), f'modality names must be unique: {names}'
        assert d_model % n_heads == 0, f'd_model ({d_model}) must be divisible by n_heads ({n_heads})'
        assert 0.0 <= p_any <= 1.0, 'p_any must be in [0, 1]'
        for c in modalities:
            assert c.mask_strategy in STRATEGIES, f'{c.name}: mask_strategy must be one of {STRATEGIES}'
            assert c.context in ('crop', 'replace'), f"{c.name}: context must be 'crop' or 'replace'"
            assert c.context == 'crop' or c.mask_input is not None, f"{c.name}: context='replace' needs mask_input"
            assert 0.0 <= c.mask_ratio <= 1.0 and 0.0 <= c.p_drop <= 1.0, f'{c.name}: probabilities must be in [0, 1]'
            assert c.dim % (c.pred_heads or pred_heads) == 0, \
                f'{c.name}: dim ({c.dim}) must be divisible by its predictor heads ({c.pred_heads or pred_heads})'
            c.encoder.eval().requires_grad_(False)
        self.cfgs, self.names, self.d, self.p_any = list(modalities), names, d_model, p_any
        self.p_drop = torch.tensor([c.p_drop for c in modalities])
        d_min = min(c.dim for c in modalities)
        self.dim_scale = [c.dim / d_min if scale_loss_by_dim else 1.0 for c in modalities]
        self.proj = nn.ModuleList(nn.Linear(c.dim, d_model) for c in modalities)
        self.mbt = MBT(len(modalities), d_model, n_layers, fusion_layer, n_bottlenecks, n_heads, mlp_dim, dropout)
        self.fusion = nn.ModuleList(nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, c.dim)) for c in modalities)
        self.mask_token = nn.ParameterList(nn.Parameter(torch.zeros(c.dim)) for c in modalities)
        self.pos = nn.ModuleList(nn.Embedding(c.max_len, c.dim) for c in modalities)
        self.pred_ln = nn.ModuleList(nn.LayerNorm(c.dim) for c in modalities)
        self.pred = nn.ModuleList(
            nn.TransformerEncoder(nn.TransformerEncoderLayer(c.dim, c.pred_heads or pred_heads,
                                                             int(pred_mlp_ratio * c.dim), pred_dropout, 'gelu',
                                                             batch_first=True, norm_first=True),
                                  pred_layers, enable_nested_tensor=False) for c in modalities)
        for t in self.mask_token:
            nn.init.trunc_normal_(t, std=0.02)
        for e in self.pos:
            nn.init.normal_(e.weight, std=0.02)

    # ---------------------------------------------------------------------------------------- frozen encoders

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        for c in self.cfgs:
            c.encoder._apply(fn)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        for c in self.cfgs:
            c.encoder.eval()
        return self

    @staticmethod
    def _encode(cfg: ModalityConfig, x, pad, pos):
        out = cfg.encoder(x, pad, pos)
        return out.last_hidden_state if hasattr(out, 'last_hidden_state') else out

    # ---------------------------------------------------------------------------------------- inputs

    def _prepare(self, inputs: Mapping[str, Mapping]):
        unknown = set(inputs) - set(self.names)
        assert not unknown, f'unknown modalities in inputs: {sorted(unknown)}'
        B = next(len(v['x']) for v in inputs.values())
        dev = self.bottleneck_device
        out = []
        for c in self.cfgs:
            v = inputs.get(c.name)
            if v is None:
                out.append(None)
                continue
            x = v['x'].to(dev)
            T = x.shape[1]
            assert T <= c.max_len, f'{c.name}: {T} tokens > max_len {c.max_len}'
            pad = v.get('pad')
            pad = torch.zeros(B, T, dtype=torch.bool, device=dev) if pad is None else pad.to(dev).bool()
            present = v.get('present')
            present = (~pad).any(1) if present is None else present.to(dev).bool() & (~pad).any(1)
            out.append((x, pad, present))
        return B, out

    @property
    def bottleneck_device(self):
        return self.mbt.bottlenecks.device

    def _present(self, B, data):
        """(B, M) bool: True where the modality is present in the data."""
        return torch.stack([torch.zeros(B, dtype=torch.bool, device=self.bottleneck_device) if d is None else d[2]
                            for d in data], 1)

    @staticmethod
    def _state(present, drop):
        return torch.where(drop & present, DROPPED, torch.where(present, PRESENT, MISSING))

    def _resolve_drop(self, B, present, drop, generator):
        dev = self.bottleneck_device
        if drop is None:
            if self.training and self.p_any > 0:
                return sample_modality_dropout(present, self.p_any, self.p_drop, generator)
            return torch.zeros(B, len(self.cfgs), dtype=torch.bool, device=dev)
        if isinstance(drop, Mapping):
            return torch.stack([drop[n].to(dev).bool() if n in drop else torch.zeros(B, dtype=torch.bool, device=dev)
                                for n in self.names], 1)
        return drop.to(dev).bool()

    # ---------------------------------------------------------------------------------------- fusion

    def _context(self, k, x, pad, present, tgt, run):
        """Projected visible tokens of modality k for the MBT: (B, T', d_model) and padding (B, T')."""
        c, B, dev = self.cfgs[k], x.shape[0], x.device
        keep = ~pad & ~tgt & present[:, None]
        r = torch.nonzero(run & keep.any(1)).squeeze(1)          # samples with >= 1 visible token in a live stream
        if not len(r):
            return (torch.zeros(B, 0, self.d, device=dev), torch.ones(B, 0, dtype=torch.bool, device=dev))
        if c.context == 'crop':
            x_c, pad_c, pos_c = crop(x[r], keep[r])
            h = self._encode(c, x_c, pad_c, pos_c)
        else:
            T = x.shape[1]
            pos = torch.arange(T, device=dev).expand(len(r), T)
            h = self._encode(c, c.mask_input(x[r], tgt[r]), pad[r], pos)
            pad_c = ~keep[r]
        T_c = h.shape[1]
        xs = torch.zeros(B, T_c, self.d, device=dev)
        xs[r] = (self.proj[k](h) * (~pad_c)[..., None]).to(xs.dtype)
        ps = torch.ones(B, T_c, dtype=torch.bool, device=dev)
        ps[r] = pad_c
        return xs, ps

    def _fuse(self, B, data, tgts, state, return_streams=False):
        xs, ps = [], []
        for k, d in enumerate(data):
            if d is None:
                xs.append(torch.zeros(B, 0, self.d, device=self.bottleneck_device))
                ps.append(torch.ones(B, 0, dtype=torch.bool, device=self.bottleneck_device))
                continue
            x, p = self._context(k, *d, tgts[k], state[:, k] == PRESENT)
            xs.append(x)
            ps.append(p)
        return self.mbt(xs, ps, state, return_streams)

    # ---------------------------------------------------------------------------------------- API

    def forward(self, inputs: Mapping[str, Mapping], targets: Mapping[str, torch.Tensor] = None, drop=None,
                generator: torch.Generator = None, return_predictions: bool = False) -> MLPAOutput:
        """One training/validation step.

        targets     optional {name: (B, T) bool} target masks; modalities not in it get masks from their mask_strategy.
        drop        None: modality dropout is sampled when the model is in training mode and p_any > 0, else none.
                    A (B, M) bool tensor or {name: (B,) bool} forces which modalities are dropped (e.g. a fixed
                    validation draw); only present modalities can be dropped.
        generator   optional CPU torch.Generator for the masks and the dropout draw (reproducible validation).
        """
        B, data = self._prepare(inputs)
        dev = self.bottleneck_device
        targets = targets or {}
        tgts = []
        for c, d in zip(self.cfgs, data):
            if d is None:
                tgts.append(None)
                continue
            x, pad, present = d
            valid = ~pad & present[:, None]
            t = targets.get(c.name)
            t = make_mask(valid, c.mask_strategy, c.mask_ratio, generator) if t is None else t.to(dev).bool()
            tgts.append(t & valid)
        present = self._present(B, data)
        state = self._state(present, self._resolve_drop(B, present, drop, generator))
        fused = self._fuse(B, data, tgts, state).mean(1)

        total, losses, preds = fused.new_zeros(()), {}, {}
        for k, (c, d, tgt) in enumerate(zip(self.cfgs, data, tgts)):
            if d is None:
                continue
            x, pad, _ = d
            r = torch.nonzero(tgt.any(1)).squeeze(1)
            if not len(r):
                continue
            t_r, T = tgt[r], tgt.shape[1]
            with torch.no_grad():
                pos = torch.arange(T, device=dev).expand(len(r), T)
                target = self._encode(c, x[r], pad[r], pos)[t_r]
            f = self.fusion[k](fused[r])[:, None]
            q = (self.mask_token[k] + self.pos[k].weight[:T]).expand(len(r), -1, -1)
            inp = self.pred_ln[k](torch.cat([f, q.to(f.dtype)], 1))
            out = self.pred[k](inp, src_key_padding_mask=torch.cat([t_r.new_zeros(len(r), 1), ~t_r], 1))[:, 1:]
            pred = out[t_r]
            loss = normalized_mse(pred, target) * self.dim_scale[k]
            losses[c.name] = loss
            total = total + c.loss_weight * loss
            if return_predictions:
                preds[c.name] = (pred, target)
        return MLPAOutput(total, losses, fused, state, dict(zip(self.names, tgts)),
                          preds if return_predictions else None)

    @torch.no_grad()
    def embed(self, inputs: Mapping[str, Mapping], return_modalities: bool = False):
        """Fused embedding (B, d_model) of the unmasked inputs, without dropout. Call model.eval() first.
        With return_modalities also {name: (B, d_model)} the mean of the modality's MBT output tokens (zero where the
        modality is missing)."""
        B, data = self._prepare(inputs)
        tgts = [None if d is None else torch.zeros_like(d[1]) for d in data]
        present = self._present(B, data)
        state = self._state(present, torch.zeros_like(present))
        b, streams = self._fuse(B, data, tgts, state, return_streams=True)
        fused = b.mean(1)
        if not return_modalities:
            return fused
        uni = {}
        for name, (r, h, p) in zip(self.names, streams):
            valid = (~p[:, 1:]).unsqueeze(-1).to(h.dtype)
            e = torch.zeros(B, self.d, device=fused.device, dtype=h.dtype)
            e[r] = (h[:, 1:] * valid).sum(1) / valid.sum(1).clamp(min=1.0)
            uni[name] = e
        return fused, uni
