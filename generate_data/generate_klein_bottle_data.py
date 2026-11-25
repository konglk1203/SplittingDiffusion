"""Gaussian mixture on the Klein bottle (paper Sec. 4.2, Fig. 4 (a)).

Each component is an isotropic Gaussian in the universal cover R^2 centred at a
random point of the fundamental polygon, folded back into [0, 2 pi)^2 with the
Klein-bottle identifications.  Output: (S, 2) array of (u, w) coordinates.

    python generate_data/generate_klein_bottle_data.py            # 10000 samples, 32 modes, seed 42 (paper)
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from manifolds.klein_bottle import KleinBottle  # noqa: E402


def generate_klein_bottle_data(num_samples=10000, num_modes=32, save_path='./dataset/klein_bottle_2d_data.npy', seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    means = torch.rand(num_modes, 2) * 2 * np.pi
    stds = torch.rand(num_modes) * 0.2 + 0.1
    counts = torch.full((num_modes,), num_samples // num_modes)
    counts[-1] += num_samples - counts.sum()
    manifold = KleinBottle()
    data = torch.cat([manifold._wrap(torch.randn(int(counts[k]), 2) * stds[k] + means[k]) for k in range(num_modes)])
    np.save(save_path, data.numpy())
    print(f"saved {save_path} {tuple(data.shape)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--num_modes", type=int, default=32)
    ap.add_argument("--out", default="./dataset/klein_bottle_2d_data.npy")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    generate_klein_bottle_data(a.n, a.num_modes, a.out, a.seed)
