import math

import torch
import torch.nn as nn


def timestep_embedding(timesteps, dim, max_period=10000):
    """Sinusoidal embedding of a (N,) tensor of (possibly fractional) times -> (N, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.get_default_dtype()) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
