"""Quick start on synthetic data: three modalities generated from one shared latent, random frozen encoders, MLPA with
modality dropout, and a linear probe of the latent from the fused embedding. Runs in about a minute on a CPU.

    python examples/quickstart_synthetic.py
"""
import torch
import torch.nn as nn

from mlpa import MLPA, ModalityConfig

torch.manual_seed(0)
N, K = 4096, 4                                   # samples, latent dim


# --------------------------------------------------------------------------------- data: 3 views of one latent z
def make_data(n):
    z = torch.randn(n, K)
    # 'codes': 16 token ids, each the argmax of a noisy random projection of z (vocab 12)
    codes = (torch.einsum('nk,tkv->ntv', z, torch.randn(16, K, 12, generator=torch.Generator().manual_seed(1)))
             + 0.5 * torch.randn(n, 16, 12)).argmax(-1)
    # 'signal': 12 time steps x 3 channels, sinusoids whose frequency and phase depend on z
    t = torch.arange(12).float()[None, :, None]
    signal = torch.sin(t * (1 + 0.3 * z[:, None, :3]) + z[:, None, 3:4]) + 0.1 * torch.randn(n, 12, 3)
    # 'tabular': 3 features, one token each
    tabular = (z @ torch.randn(K, 3, generator=torch.Generator().manual_seed(2)) + 0.1 * torch.randn(n, 3))[..., None]
    # 'codes' is missing for 20% of the samples, 'signal' has variable length (right padding)
    present_codes = torch.rand(n) > 0.2
    lengths = torch.randint(6, 13, (n,))
    pad_signal = torch.arange(12)[None] >= lengths[:, None]
    inputs = {'codes': {'x': codes, 'present': present_codes},
              'signal': {'x': signal, 'pad': pad_signal},
              'tabular': {'x': tabular}}
    return inputs, z


def subset(inputs, idx):
    return {m: {k: v[idx] for k, v in d.items()} for m, d in inputs.items()}


# --------------------------------------------------------------------------------- frozen uni-modal encoders
class Encoder(nn.Module):
    """Any module with forward(x, pad, pos) -> (B, T, dim) works. Here: a random, untrained tiny transformer."""

    def __init__(self, dim, max_len, vocab=None, in_features=None):
        super().__init__()
        self.inp = nn.Embedding(vocab, dim) if vocab else nn.Linear(in_features, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.body = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, 4, 2 * dim, 0.0, batch_first=True), 2,
                                          enable_nested_tensor=False)

    def forward(self, x, pad, pos):
        return self.body(self.inp(x) + self.pos(pos), src_key_padding_mask=pad)


modalities = [
    ModalityConfig('codes', Encoder(32, 16, vocab=12), dim=32, max_len=16, mask_ratio=0.5, mask_strategy='block',
                   p_drop=0.3),
    ModalityConfig('signal', Encoder(48, 12, in_features=3), dim=48, max_len=12, mask_ratio=0.4, p_drop=0.3),
    ModalityConfig('tabular', Encoder(16, 3, in_features=1), dim=16, max_len=3, mask_ratio=0.34,
                   mask_strategy='bernoulli', p_drop=0.1, loss_weight=0.5),
]
model = MLPA(modalities, d_model=64, n_layers=4, n_bottlenecks=4, n_heads=4, mlp_dim=128, p_any=0.5)
print(f'trainable parameters: {sum(p.numel() for p in model.parameters()):,}')

inputs, z = make_data(N)
train_idx, test_idx = torch.arange(0, 3072), torch.arange(3072, N)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

for epoch in range(8):
    model.train()
    perm = train_idx[torch.randperm(len(train_idx))]
    tot = 0.0
    for i in range(0, len(perm), 128):
        out = model(subset(inputs, perm[i:i + 128]))
        opt.zero_grad(set_to_none=True)          # a stream dropped in a whole batch keeps grad None
        out.loss.backward()
        opt.step()
        tot += out.loss.item()
    per = ' '.join(f'{k} {v.item():.3f}' for k, v in out.losses.items())
    print(f'epoch {epoch} | train loss {tot / (len(perm) / 128):.4f} | last batch {per}')

# --------------------------------------------------------------------------------- linear probe of z
model.eval()
emb = model.embed(inputs)                        # (N, d_model)


def probe_r2(e):
    X = torch.cat([e, torch.ones(len(e), 1)], 1)
    w = torch.linalg.lstsq(X[train_idx], z[train_idx]).solution
    resid = z[test_idx] - X[test_idx] @ w
    return 1 - resid.pow(2).sum() / (z[test_idx] - z[test_idx].mean(0)).pow(2).sum()


print(f'linear probe R^2 of z from the fused embedding: {probe_r2(emb):.3f}')
without_codes = {k: v for k, v in inputs.items() if k != 'codes'}
print(f'... with the codes modality missing everywhere:  {probe_r2(model.embed(without_codes)):.3f}')
