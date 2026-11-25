"""Sample-based energy distance for QM9 molecules on St(n, 4) (paper Sec. 4.4,
Tab. 4; definition in App. C).

NLL on St(n,4) is biased by the inference discretisation (paper Tab. 3), so
sample quality is measured with a likelihood-free criterion.  Every molecule
X in St(n,4) (columns 0-2 = mass-weighted coordinates r_i, column 3 = sqrt(m_i/M))
is mapped to four descriptor vectors that are invariant to atom permutation,
global O(3) and translation:

  pdist       sorted pairwise distances ||r_i - r_j||
  mass        sorted fourth column
  massw_dist  sorted ||r_i - r_j|| sqrt(mu_i mu_j)
  shape       radius of gyration and the three sorted eigenvalues of (1/n) sum_i r_i r_i^T

Each descriptor is standardised with the real-data statistics and compared
between generated and real molecules with the energy distance
ED(A, B) = 2 E|a-b| - E|a-a'| - E|b-b'| (U-statistic over up to --max_pts points
per set).  TOTAL is the sum over the four descriptors; FLOOR is the same
quantity between two disjoint halves of the real data (finite-sample noise).

    python scripts/qm9_sample_metric.py --data dataset/qm9_by_size/qm9_N020.npy --N 20 --n 2000 \
        --gen splitting=samples/sp20.npy --gen rfm=samples/rfm20.npy

``--split heldout`` restricts the real molecules to the 10% validation split of
main.py (same --train_seed); the default 'all' draws them from the whole file,
as in the reported numbers.
"""
import argparse

import numpy as np
from sklearn.model_selection import train_test_split

EPS = 1e-9


def load_mols(path, N):
    X = np.load(path).astype(np.float64).reshape(-1, N, 4)
    return X[np.isfinite(X).all(axis=(1, 2))]          # drop failed (non-finite) samples


def descriptors(X):
    """X: (B, N, 4) -> dict of (B, D) permutation / O(3) / translation invariant descriptors."""
    B, N, _ = X.shape
    pos = X[:, :, :3]
    pos = pos - pos.mean(axis=1, keepdims=True)
    mass = X[:, :, 3]
    iu, ju = np.triu_indices(N, k=1)
    pdist = np.sqrt(((pos[:, iu, :] - pos[:, ju, :]) ** 2).sum(-1) + EPS)
    # generated points may have a negative mass column (its sign is free on St(n,4)); clip for the sqrt
    mw = pdist * np.sqrt(np.clip(mass[:, iu] * mass[:, ju], 0.0, None) + EPS)
    rg = np.sqrt((pos ** 2).sum(-1).mean(1) + EPS)[:, None]
    evals = np.linalg.eigvalsh(np.einsum('bni,bnj->bij', pos, pos) / N)
    return {
        'pdist': np.sort(pdist, axis=1),
        'mass': np.sort(mass, axis=1),
        'massw_dist': np.sort(mw, axis=1),
        'shape': np.concatenate([rg, evals], axis=1),
    }


def energy_distance(A, B, max_pts=1500, seed=0):
    rng = np.random.RandomState(seed)
    if A.shape[0] > max_pts:
        A = A[rng.choice(A.shape[0], max_pts, replace=False)]
    if B.shape[0] > max_pts:
        B = B[rng.choice(B.shape[0], max_pts, replace=False)]

    def mean_dist(P, Q):
        return np.sqrt(np.maximum(((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1), 0.0) + EPS).mean()

    return 2 * mean_dist(A, B) - mean_dist(A, A) - mean_dist(B, B)


def standardize(ref, *others):
    mu, sd = ref.mean(0, keepdims=True), ref.std(0, keepdims=True) + 1e-8
    return [(x - mu) / sd for x in (ref,) + others]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='real molecules, dataset/qm9_by_size/qm9_N<n>.npy')
    ap.add_argument('--gen', action='append', default=[], help='generated .npy, repeatable; "name=path" or "path"')
    ap.add_argument('--N', type=int, required=True, help='atom count n')
    ap.add_argument('--n', type=int, default=2000, help='molecules per set')
    ap.add_argument('--max_pts', type=int, default=1500, help='points per set in the energy-distance U-statistic')
    ap.add_argument('--split', choices=['all', 'heldout'], default='all')
    ap.add_argument('--train_seed', type=int, default=1, help='seed of the main.py 90/10 split (for --split heldout)')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    rng = np.random.RandomState(a.seed)
    data = load_mols(a.data, a.N)
    if a.split == 'heldout':
        _, data = train_test_split(data, test_size=0.1, random_state=a.train_seed)
    idx = rng.permutation(len(data))
    half = min(a.n, len(data) // 2)
    desc_a, desc_b = descriptors(data[idx[:half]]), descriptors(data[idx[half:2 * half]])
    keys = list(desc_a)

    def row(name, desc, n_mol):
        vals = {}
        for k in keys:
            ref_s, other_s = standardize(desc_a[k], desc[k])
            vals[k] = energy_distance(other_s, ref_s, max_pts=a.max_pts, seed=a.seed)
        print(f"{name[:28]:28s} | " + " | ".join(f"{vals[k]:11.4f}" for k in keys)
              + f" | {sum(vals.values()):8.4f}   (n_mol={n_mol})")

    print(f"real molecules: {len(data)} ({a.split}); reference = {half}, floor from a disjoint half of {half}")
    header = f"{'set':28s} | " + " | ".join(f"{k:>11s}" for k in keys) + " |   TOTAL"
    print(header)
    print("-" * len(header))
    row("FLOOR (real vs real)", desc_b, half)
    for spec in a.gen:
        name, path = spec.split('=', 1) if '=' in spec else (spec, spec)
        g = load_mols(path, a.N)
        if len(g) > a.n:
            g = g[rng.permutation(len(g))[:a.n]]
        row(name, descriptors(g), len(g))


if __name__ == '__main__':
    main()
