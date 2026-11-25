"""
QM9 conformations as points of the Stiefel manifold St(N, 4), one dataset per
atom count N (paper Sec. 4.4, Tab. 4 "# Data").

Input: the processed QM9 splits ``qm9_{train,val,test}.pyg`` of Stiefel Flow
Matching (Cheng et al., 2024, https://github.com/the-matter-lab/stiefelFM),
which already contain the Cartesian coordinates, atomic numbers and principal
moments of inertia of every conformer.  All three splits are pooled; the 90/10
train/validation split used for training is done in ``main.py``.

Output: ``<out_dir>/qm9_N<N>.npy`` of shape (num_molecules, N, 4) with rows
on St(N, 4):
  cols 0-2: sqrt(m_i / I_a) * r_{ia}   (mass-weighted coordinates along the principal axes)
  col  3:   sqrt(m_i / M)              (normalised square-root mass)

Usage:
    python generate_data/preprocess_qm9_by_size.py --data_dir ./dataset --out_dir ./dataset/qm9_by_size
"""

import sys
import types
import argparse
import collections
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Atom mass table (indexed by atomic number)
# ---------------------------------------------------------------------------
_ATOM_MASSES = np.array([
    0.0, 1.00782503223, 4.00260325413, 7.0160034366, 9.012183065, 11.00930536,
    12.0, 14.00307400443, 15.99491461957, 18.99840316273, 19.9924401762,
    22.9897692820, 23.985041697, 26.98153853, 27.97692653465, 30.97376199842,
    31.9720711744, 34.968852682, 39.9623831237, 38.9637064864, 39.962590863,
    44.95590828, 47.94794198, 50.9439595, 51.9405075, 54.93804391,
    55.93493633, 58.93319429, 57.93534241, 62.92959772, 63.92914201,
    68.9255735, 73.921177761, 74.92159457, 79.9165218, 78.9183376,
    83.9114977282, 84.9117897379, 87.9056125, 88.9058403, 89.9046977,
    92.9063730, 97.9072124, 98.0, 101.9032056, 102.9054980, 105.907652,
    106.9050916, 113.903365, 112.9040618, 119.9022016, 120.9038120,
    129.90622275, 126.9044719, 131.90415509, 132.90545196, 137.9052470,
    138.9063563, 139.9054431, 140.9076576, 144.9127559, 151.9197397,
    152.9212380, 157.9241123, 158.9253547, 162.930039, 164.9303288,
    167.933208, 168.934219, 173.9388664, 174.9407752, 179.946557,
    180.9479958, 183.95093092, 186.9557501, 191.961477, 192.962921,
    195.964569, 196.9665687, 201.970643, 204.974427, 207.9766521,
    208.9803991, 208.9824308, 209.9871479, 222.0175782, 223.0197360,
    226.0254103, 227.0277523, 232.0380558, 231.0358842, 238.0507884,
], dtype=np.float64)


# ---------------------------------------------------------------------------
# PyG pickle shims
# ---------------------------------------------------------------------------
def _ensure_pyg_pickle_shims():
    """Inject stub modules so torch.load can unpickle old PyG files."""
    for mod_name in [
        'torch_geometric',
        'torch_geometric.data',
        'torch_geometric.data.data',
        'torch_geometric.data.storage',
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    for mod_name, class_names in [
        ('torch_geometric.data',         ('Data',)),
        ('torch_geometric.data.data',    ('DataEdgeAttr', 'DataTensorAttr')),
        ('torch_geometric.data.storage', ('BaseStorage', 'NodeStorage',
                                          'EdgeStorage', 'GlobalStorage', 'TensorAttr')),
    ]:
        mod = sys.modules[mod_name]
        for cls in class_names:
            if not hasattr(mod, cls):
                setattr(mod, cls, type(cls, (), {}))

    for mod_name in ['src', 'src.datasets', 'src.datasets.dataset_pyg', 'src.chem']:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)
    if not hasattr(sys.modules['src.datasets.dataset_pyg'], 'GraphData'):
        sys.modules['src.datasets.dataset_pyg'].GraphData = type('GraphData', (), {})
    sys.modules['src'].datasets        = sys.modules['src.datasets']
    sys.modules['src.datasets'].dataset_pyg = sys.modules['src.datasets.dataset_pyg']
    if not hasattr(sys.modules['src.chem'], 'Molecule'):
        sys.modules['src.chem'].Molecule = type('Molecule', (), {})
    sys.modules['src'].chem = sys.modules['src.chem']


# ---------------------------------------------------------------------------
# Load & convert one .pyg split
# ---------------------------------------------------------------------------
def load_pyg(pyg_path: Path):
    _ensure_pyg_pickle_shims()
    try:
        payload = torch.load(pyg_path, map_location='cpu', weights_only=False)
    except TypeError:
        payload = torch.load(pyg_path, map_location='cpu')

    data, slices = payload
    mapping   = data._store._mapping
    n_samples = len(slices[next(iter(slices))]) - 1
    return mapping, slices, n_samples


def mol_to_stiefel(xyz, atoms, moments):
    """
    Convert one molecule to a point on St(N, 4).

    Args:
        xyz     : (N, 3) float64 Cartesian coordinates
        atoms   : (N,)   int64   atomic numbers
        moments : (3,)   float64 principal moments of inertia

    Returns:
        (N, 4) float32 array on St(N, 4)
    """
    masses     = _ATOM_MASSES[atoms][:, None]           # (N, 1)
    masses_sum = masses.sum()
    coords_c   = xyz - (masses * xyz).sum(0, keepdims=True) / masses_sum
    onb_3      = coords_c * np.sqrt(masses / moments[None, :])
    onb_4      = np.concatenate([onb_3, np.sqrt(masses / masses_sum)], axis=-1)
    return onb_4.astype(np.float32)                     # (N, 4)


def convert_split(pyg_path: Path, split_name: str):
    """
    Load one .pyg split and return a dict  {N: list_of_(N,4)_arrays}.
    """
    print(f'  Loading {pyg_path} ...', flush=True)
    mapping, slices, n_samples = load_pyg(pyg_path)

    sv_c = slices['coords']
    sv_a = slices['atoms']
    sv_m = slices['moments']

    # groups[N] = list of (N, 4) float32 arrays
    groups: dict[int, list] = collections.defaultdict(list)
    skipped = 0

    for i in range(n_samples):
        xyz = mapping['coords'][int(sv_c[i]):int(sv_c[i+1])].numpy().astype(np.float64)
        N   = xyz.shape[0]

        atoms   = mapping['atoms'][int(sv_a[i]):int(sv_a[i+1])].numpy().astype(np.int64).reshape(-1)
        moments = mapping['moments'][int(sv_m[i]):int(sv_m[i+1])].numpy().astype(np.float64).reshape(-1)[:3]

        # Skip degenerate molecules (zero principal moment → can't normalise)
        if (moments <= 0).any():
            skipped += 1
            continue

        point = mol_to_stiefel(xyz, atoms, moments)     # (N, 4)
        groups[N].append(point)

    print(f'  {split_name}: {n_samples} total, {skipped} skipped, '
          f'{sum(len(v) for v in groups.values())} converted, '
          f'{len(groups)} unique sizes: {sorted(groups.keys())}')
    return groups


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_dir', default='./dataset',
                        help='Directory containing qm9_train.pyg etc.')
    parser.add_argument('--out_dir',  default='./dataset/qm9_by_size',
                        help='Output directory for per-size .npy files')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        choices=['train', 'val', 'test'],
                        help='Which splits to process (default: all three)')
    parser.add_argument('--verify_ortho', action='store_true',
                        help='Print orthogonality residuals after conversion')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Accumulate across splits: combined[N] = list of (N,4) arrays
    combined: dict[int, list] = collections.defaultdict(list)

    for split in args.splits:
        pyg_path = data_dir / f'qm9_{split}.pyg'
        if not pyg_path.exists():
            print(f'  [skip] {pyg_path} not found')
            continue
        groups = convert_split(pyg_path, split)
        for N, pts in groups.items():
            combined[N].extend(pts)

    # Save one file per atom count
    print(f'\nSaving to {out_dir} ...')
    summary = []
    for N in sorted(combined.keys()):
        arr = np.stack(combined[N])          # (count, N, 4)
        out_path = out_dir / f'qm9_N{N:03d}.npy'
        np.save(out_path, arr)

        # Optional orthogonality check
        orth_str = ''
        if args.verify_ortho:
            sub  = arr[:min(500, len(arr))].astype(np.float64)
            XtX  = np.einsum('bni,bnj->bij', sub, sub)
            err  = np.linalg.norm(XtX - np.eye(4), axis=(-2, -1)).max()
            orth_str = f'  ortho_err={err:.2e}'

        summary.append((N, len(combined[N]), out_path.name))
        print(f'  N={N:3d}  count={len(combined[N]):6d}  → {out_path.name}{orth_str}')

    print(f'\nDone. {len(summary)} files written to {out_dir}')


if __name__ == '__main__':
    main()
