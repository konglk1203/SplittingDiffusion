"""Wrapped-normal mixture on the hyperbolic plane H^2 (paper Sec. 4.2, Fig. 4 (c)).

Four components whose centres mu_k = exp_o(c_k) sit at geodesic distance
``radius`` from the origin, equally spaced in angle; each sample is
x = exp_{mu_k}(PT_{o -> mu_k}(scale * N(0, I_2))).  Output: (S, 3) points of the
Lorentz hyperboloid.

    python generate_data/generate_h2_data.py            # 100000 samples, radius 2, scale 0.3, seed 0 (paper)
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from manifolds.hyperbolic import Hyperbolic, _mink_dot  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--out", default="./dataset/h2_wgmm.npy")
    ap.add_argument("--radius", type=float, default=2.0, help="geodesic distance of the component centres from the origin")
    ap.add_argument("--scale", type=float, default=0.3, help="std of each wrapped normal")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    torch.set_default_dtype(torch.float64)
    m = Hyperbolic(2)
    o = m.origin(num=1)
    K = 4
    centers = []
    for ang in torch.arange(K) * (2 * np.pi / K):
        c_tan = torch.zeros(1, 3)
        c_tan[0, 1], c_tan[0, 2] = a.radius * torch.cos(ang), a.radius * torch.sin(ang)
        centers.append(m.exp_map(o, c_tan))
    centers = torch.cat(centers, 0)

    per = a.n // K
    o_b = m.origin(num=per)
    pts = []
    for k in range(K):
        mu = centers[k:k + 1].expand(per, 3).contiguous()
        tan_o = torch.zeros(per, 3)
        tan_o[:, 1:] = torch.randn(per, 2) * a.scale
        pts.append(m.exp_map(mu, m._parallel_transport(o_b, mu, tan_o)))
    X = torch.cat(pts, 0)
    X = X[torch.randperm(X.shape[0])]

    err = (_mink_dot(X, X).squeeze(-1) + 1).abs().max().item()
    print(f"generated {tuple(X.shape)}; on-manifold error {err:.2e}; mean distance to o {m._dist_to_origin(X).mean().item():.3f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.save(a.out, X.float().numpy())
    print("saved", a.out)


if __name__ == "__main__":
    main()
