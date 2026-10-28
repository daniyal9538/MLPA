import pytest
import torch
import torch.nn as nn

from mlpa import (DROPPED, MISSING, MLPA, PRESENT, ModalityConfig, block_mask, bernoulli_mask, crop,
                  random_mask, sample_modality_dropout)


class TinyEncoder(nn.Module):
    """Token embedding (ids or features) + position embedding + one transformer layer."""

    def __init__(self, dim, vocab=None, in_features=None, max_len=32):
        super().__init__()
        self.inp = nn.Embedding(vocab, dim) if vocab else nn.Linear(in_features, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.layer = nn.TransformerEncoderLayer(dim, 2, 2 * dim, 0.0, batch_first=True)

    def forward(self, x, pad, pos):
        return self.layer(self.inp(x) + self.pos(pos), src_key_padding_mask=pad)


def make_model(**kw):
    mods = [ModalityConfig('ids', TinyEncoder(16, vocab=20), 16, 12),
            ModalityConfig('feat', TinyEncoder(32, in_features=5), 32, 8, mask_strategy='block'),
            ModalityConfig('tab', TinyEncoder(8, in_features=1, max_len=3), 8, 3, mask_strategy='bernoulli',
                           mask_ratio=0.4)]
    return MLPA(mods, d_model=32, n_layers=2, n_bottlenecks=3, **kw)


def make_inputs(B=6):
    pad = torch.zeros(B, 12, dtype=torch.bool)
    pad[0, 6:] = True
    return {'ids': {'x': torch.randint(0, 20, (B, 12)), 'pad': pad},
            'feat': {'x': torch.randn(B, 8, 5), 'present': torch.tensor([1, 0, 1, 1, 1, 0] * (B // 6)).bool()},
            'tab': {'x': torch.randn(B, 3, 1)}}


# ------------------------------------------------------------------------------------------------ masking

@pytest.mark.parametrize('fn', [random_mask, bernoulli_mask, block_mask])
def test_masks_only_valid_and_keep_context(fn):
    g = torch.Generator().manual_seed(0)
    valid = torch.ones(50, 10, dtype=torch.bool)
    valid[:, 7:] = False
    valid[0, 1:] = False                                      # single token
    valid[1] = False                                          # empty row
    for _ in range(20):
        t = fn(valid, 0.5, g)
        assert not (t & ~valid).any()
        n_t, n_v = t.sum(1), valid.sum(1)
        assert (n_t[n_v > 0] >= 1).all() and n_t[1] == 0
        assert ((n_v - n_t)[n_v >= 2] >= 1).all()


def test_random_mask_count():
    valid = torch.ones(4, 10, dtype=torch.bool)
    assert (random_mask(valid, 0.3).sum(1) == 3).all()


def test_block_mask_contiguous():
    t = block_mask(torch.ones(30, 16, dtype=torch.bool), 0.25, torch.Generator().manual_seed(1))
    for row in t:
        idx = row.nonzero().squeeze(1)
        assert len(idx) == 4 and (idx.diff() == 1).all()


def test_crop():
    x = torch.arange(12).reshape(2, 6, 1).float()
    keep = torch.tensor([[1, 0, 1, 0, 0, 1], [0, 1, 0, 0, 0, 0]]).bool()
    xc, pad, pos = crop(x, keep)
    assert xc.shape == (2, 3, 1)
    assert xc[0, :, 0].tolist() == [0, 2, 5] and pos[0].tolist() == [0, 2, 5]
    assert xc[1, 0, 0] == 7 and pad[1].tolist() == [False, True, True] and xc[1, 1:].abs().sum() == 0


# ------------------------------------------------------------------------------------------------ dropout

def test_dropout_rules():
    g = torch.Generator().manual_seed(0)
    avail = torch.rand(2000, 4, generator=g) < 0.7
    p = torch.tensor([0.5, 0.5, 0.0, 0.9])
    drop = sample_modality_dropout(avail, 1.0, p, g)
    assert not (drop & ~avail).any()                          # missing never dropped
    assert not drop[:, 2].any()                               # p_drop 0 never dropped
    multi = avail.sum(1) >= 2
    assert not drop[~multi].any()                             # single modality never dropped
    assert ((avail & ~drop).sum(1)[avail.any(1)] >= 1).all()  # one always stays
    sel = multi & avail[:, [0, 1, 3]].any(1)
    assert drop[sel].any(1).all()                             # p_any = 1: every eligible sample loses one


def test_dropout_p_any_zero():
    assert not sample_modality_dropout(torch.ones(10, 3, dtype=torch.bool), 0.0, torch.ones(3)).any()


# ------------------------------------------------------------------------------------------------ model

def test_forward_backward_and_states():
    torch.manual_seed(0)
    model = make_model(p_any=1.0)
    out = model(make_inputs())
    assert out.fused.shape == (6, 32) and torch.isfinite(out.loss)
    assert set(out.losses) == {'ids', 'feat', 'tab'}
    assert (out.state[[1, 5], 1] == MISSING).all()
    assert (out.state == DROPPED).any(1).all()                # p_any = 1 and every row has >= 2 modalities
    out.loss.backward()
    assert all(p.grad is None for c in model.cfgs for p in c.encoder.parameters())


def test_eval_no_dropout_and_missing_modality():
    model = make_model(p_any=1.0).eval()
    inputs = make_inputs()
    del inputs['tab']
    out = model(inputs)
    assert not (out.state == DROPPED).any() and (out.state[:, 2] == MISSING).all() and 'tab' not in out.losses


def test_forced_drop_and_custom_targets():
    model = make_model()
    tgt = torch.zeros(6, 12, dtype=torch.bool)
    tgt[:, 2] = True
    out = model(make_inputs(), targets={'ids': tgt}, drop={'ids': torch.ones(6, dtype=torch.bool)},
                return_predictions=True)
    assert (out.state[:, 0] == DROPPED).all() and out.targets['ids'].sum() == 6
    pred, target = out.predictions['ids']
    assert pred.shape == target.shape == (6, 16)


def test_dropped_stream_gets_no_gradient():
    model = make_model()
    out = model(make_inputs(), drop={'ids': torch.ones(6, dtype=torch.bool)})
    out.loss.backward()
    assert all(p.grad is None for p in model.mbt.streams[0].parameters())
    assert model.proj[0].weight.grad is None
    assert model.pred[0].layers[0].linear1.weight.grad is not None   # its predictor still learns


def test_replace_context_and_embed():
    enc = TinyEncoder(16, vocab=21)
    mask_input = lambda x, t: x.masked_fill(t, 20)
    model = MLPA([ModalityConfig('a', enc, 16, 10, context='replace', mask_input=mask_input),
                  ModalityConfig('b', TinyEncoder(16, in_features=3), 16, 10)], d_model=16, n_layers=3,
                 fusion_layer=1)
    inputs = {'a': {'x': torch.randint(0, 20, (4, 10))}, 'b': {'x': torch.randn(4, 10, 3)}}
    assert torch.isfinite(model(inputs).loss)
    model.eval()
    fused, uni = model.embed(inputs, return_modalities=True)
    assert fused.shape == (4, 16) and uni['a'].shape == (4, 16)
    inputs['b']['present'] = torch.tensor([True, False, True, True])
    _, uni = model.embed(inputs, return_modalities=True)
    assert uni['b'][1].abs().sum() == 0


def test_state_dict_excludes_encoders():
    model = make_model()
    assert not any(k.startswith('cfgs') or 'encoder.' in k and 'pred' not in k for k in model.state_dict())
    n_enc = sum(p.numel() for c in model.cfgs for p in c.encoder.parameters())
    assert n_enc > 0 and all(p.requires_grad for p in model.parameters())


def test_validation_errors():
    with pytest.raises(AssertionError):
        MLPA([ModalityConfig('a', TinyEncoder(10, vocab=5), 10, 4)], pred_heads=3)
    with pytest.raises(AssertionError):
        MLPA([ModalityConfig('a', TinyEncoder(8, vocab=5), 8, 4, context='replace')])
