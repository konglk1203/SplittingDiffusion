"""Transformer (DiT-style, AdaLN-Zero conditioning) score network for the
Stiefel manifold St(n, p), used for the QM9 experiment (paper Sec. 4.4,
Tab. 7: hidden 512, 5 blocks, 8 heads).

Tokenisation is row-wise: token i is row i of the n x p matrix x (p features),
concatenated with row i of v (another p features) for splitting diffusion.  The
output is one p-vector per token, i.e. an n x p matrix in the same layout as
``Stiefel.vectorize_v``.  Learnable positional embeddings make the network
aware of the atom index.
"""
import torch
import torch.nn as nn

from models.utils import timestep_embedding


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, hidden_dim))
        # (shift, scale, gate) for attention and for the MLP, zero-initialised
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim, bias=True))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c):
        # x: (B, n_tokens, hidden)   c: (B, hidden)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            [u.unsqueeze(1) for u in self.adaLN(c).chunk(6, dim=-1)]
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.attn(h, h, h)[0]
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        return x + gate_mlp * self.mlp(h)


class StiefelDiT(nn.Module):
    def __init__(self, n, p, hidden_dim, num_heads, num_layers, with_velocity, time_embed_dim=256, v_sigma=1.0):
        super().__init__()
        self.n, self.p = n, p
        self.with_velocity = with_velocity
        # v is divided by its prior std so the network sees O(1) features for every n.
        self.register_buffer('v_sigma', torch.tensor(float(v_sigma)))
        self.token_embed = nn.Linear(2 * p if with_velocity else p, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, n, hidden_dim))
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True))
        self.final_proj = nn.Linear(hidden_dim, p)
        for m in (self.final_proj, self.final_adaLN[-1]):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, v, t):
        bs, n, p = x.shape[0], self.n, self.p
        tokens = x.view(bs, n, p)
        if v is not None:
            tokens = torch.cat([tokens, (v / self.v_sigma).view(bs, n, p)], dim=-1)
        tokens = self.token_embed(tokens) + self.pos_embed
        c = self.time_embed(timestep_embedding(t, self.time_embed[0].in_features))
        for block in self.blocks:
            tokens = block(tokens, c)
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        tokens = modulate(self.final_norm(tokens), shift.unsqueeze(1), scale.unsqueeze(1))
        return self.final_proj(tokens).reshape(bs, -1)


def build_dit(opt, input_dim, out_dim, manifold):
    from manifolds.stiefel import Stiefel
    if not isinstance(manifold, Stiefel):
        raise ValueError(f"net_name=dit is implemented for the Stiefel manifold only (got {type(manifold).__name__})")
    n, p = manifold.n, manifold.p
    assert out_dim == n * p
    hidden_dim = opt.model_hidden_dim
    num_heads = opt.model_num_heads or max(1, hidden_dim // 64)
    if hidden_dim % num_heads != 0:
        num_heads = max(h for h in range(1, num_heads + 1) if hidden_dim % h == 0)
    return StiefelDiT(n, p, hidden_dim, num_heads, opt.model_blocks,
                      with_velocity=(input_dim == 2 * out_dim), v_sigma=opt.velocity_sigma)
