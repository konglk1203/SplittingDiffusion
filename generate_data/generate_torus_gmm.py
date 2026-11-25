"""Gaussian mixture on the flat torus T^2 (paper Sec. 4.1, Fig. 3).

Writes two files:
  <out>.npy          (S, 2) angles in [-pi, pi)
  <out>_params.npz   mus (K,2), covs (K,2,2), weights (K,)  -- the mixture
                     parameters, needed by the analytic checks in analysis/
                     (SplittingTargetKernelT2 evaluates the exact score of the
                     split chain started from this mixture).

Default: K = 7 components with random means, random anisotropic covariances
and random weights.  (The random draw used for the paper's figures is not
reproduced bit-for-bit; any mixture works for the checks.)

    python generate_data/generate_torus_gmm.py --out dataset/torus_gmm_t2
"""
import argparse
import os

import numpy as np

TWO_PI = 2 * np.pi


def generate_torus_gmm(num_samples=100000, num_modes=7, std_range=(0.15, 0.45), seed=0):
    rng = np.random.default_rng(seed)
    mus = rng.uniform(-np.pi, np.pi, size=(num_modes, 2))
    covs = np.empty((num_modes, 2, 2))
    for k in range(num_modes):
        stds = rng.uniform(*std_range, size=2)
        theta = rng.uniform(0, np.pi)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        covs[k] = R @ np.diag(stds ** 2) @ R.T
    weights = rng.uniform(0.5, 1.5, size=num_modes)
    weights /= weights.sum()

    comp = rng.choice(num_modes, size=num_samples, p=weights)
    data = np.empty((num_samples, 2))
    for k in range(num_modes):
        idx = comp == k
        data[idx] = rng.multivariate_normal(mus[k], covs[k], size=idx.sum())
    data = (data + np.pi) % TWO_PI - np.pi
    return data.astype(np.float32), dict(mus=mus, covs=covs, weights=weights)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./dataset/torus_gmm_t2", help="output prefix (.npy and _params.npz are appended)")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--num_modes", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    data, params = generate_torus_gmm(a.n, a.num_modes, seed=a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.save(a.out + ".npy", data)
    np.savez(a.out + "_params.npz", **params)
    print(f"saved {a.out}.npy {data.shape} and {a.out}_params.npz ({a.num_modes} components)")
