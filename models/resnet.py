"""Residual MLP score network used for the low-dimensional manifolds
(T^2, Klein bottle, RP^2, H^2, SO(3)).

Input: the flat vectors x (and v for splitting) concatenated; the time t enters
through a sinusoidal embedding added to the hidden features.
"""
import numpy as np
import torch
import torch.nn as nn

from models.utils import SiLU, timestep_embedding


class FCnet(nn.Module):
    def __init__(self, data_dim, hidden_dim, num_res_blocks):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.map = nn.Linear(data_dim, hidden_dim)
        self.norm1 = nn.GroupNorm(32, hidden_dim)   # unused in forward(); kept so older checkpoints load
        self.res_blocks = nn.ModuleList([self._build_res_block() for _ in range(num_res_blocks)])
        self.norms = nn.ModuleList([nn.GroupNorm(32, hidden_dim) for _ in range(num_res_blocks)])

    def _build_res_block(self):
        hid = self.hidden_dim
        layers = []
        for _ in range(3):
            layers += [nn.Linear(hid, hid), SiLU()]
        return nn.Sequential(*layers)

    def forward(self, x):
        h = self.map(x)
        for res_block, norm in zip(self.res_blocks, self.norms):
            h = (h + res_block(norm(h))) / np.sqrt(2)
        return h


class ResNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, time_embed_dim=128, num_res_blocks=6):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.t_module = nn.Sequential(nn.Linear(time_embed_dim, hidden_dim), SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.x_module = FCnet(data_dim=input_dim, hidden_dim=hidden_dim, num_res_blocks=num_res_blocks)
        self.norm = nn.GroupNorm(32, hidden_dim)
        self.out_module = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), SiLU(),
            nn.Linear(hidden_dim, hidden_dim), SiLU(),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, x, v, t):
        u = x if v is None else torch.cat([x, v], dim=-1)
        t_out = self.t_module(timestep_embedding(t, self.time_embed_dim))
        return self.out_module(self.norm(self.x_module(u) + t_out))


def build_resnet(opt, input_dim, out_dim):
    return ResNet(input_dim, out_dim, hidden_dim=opt.model_hidden_dim, num_res_blocks=opt.model_blocks)
