"""Wrapped-Gaussian mixture on RP^2 (paper Sec. 4.2, Fig. 4 (b)).

Component centres are uniform on RP^2; each component is a tangent Gaussian at
its centre pushed to the manifold with the exponential map (the same
``geodesic_integrator`` used for training).  Output: (S, 3) points of the
upper-hemisphere fundamental domain.

    python generate_data/generate_rp2_data.py            # 10000 samples, 8 modes, std 0.2, seed 42 (paper)
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('GEOMSTATS_BACKEND', 'pytorch')
from manifolds.real_projective_plane import RealProjectivePlane  # noqa: E402


def generate_rp2_data(num_samples=10000, num_modes=8, std=0.2, save_path='./dataset/rp2_gmm_data.npy', seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    manifold = RealProjectivePlane()
    centers, _ = manifold.sample_prior_vector(num_modes)
    counts = np.full(num_modes, num_samples // num_modes)
    counts[-1] += num_samples - counts.sum()
    data = []
    for k in range(num_modes):
        center = centers[k:k + 1].repeat(int(counts[k]), 1)
        noise = manifold.project_to_tangent_space(center, torch.randn(int(counts[k]), 3) * std)
        x, _ = manifold.geodesic_integrator(center, noise, alpha=1.0)
        data.append(x)
    data = torch.cat(data)
    np.save(save_path, data.cpu().numpy())
    print(f"saved {save_path} {tuple(data.shape)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--num_modes", type=int, default=8)
    ap.add_argument("--std", type=float, default=0.2)
    ap.add_argument("--out", default="./dataset/rp2_gmm_data.npy")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    generate_rp2_data(a.n, a.num_modes, a.std, a.out, a.seed)
