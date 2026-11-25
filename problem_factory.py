"""Build the manifold, the dataset and the Lightning runner for a problem.

Problems (paper Sec. 4):

================  ===========================  ======================================
problem_name      manifold                     paper
================  ===========================  ======================================
``torus``         flat torus T^d               Sec. 4.1 (T^2 GMM; quick-start example)
``klein_bottle``  Klein bottle K^2             Sec. 4.2, Fig. 4 (a, d)
``rp2``           real projective plane RP^2   Sec. 4.2, Fig. 4 (b, e)
``hyperbolic``    hyperbolic plane H^2         Sec. 4.2, Fig. 4 (c, f)
``so3``           SO(3)                        Sec. 4.3, Tab. 2
``qm9``           Stiefel St(n, 4)             Sec. 4.4, Tab. 3-4; App. B-C
================  ===========================  ======================================

``opt.nll_offset`` is log Vol(M): on a compact manifold the stationary position
law is uniform, so log p_T(x) = -nll_offset (used by the NLL of Thm. 3).  On
H^2 the stationary law is the wrapped normal exp(-U)/Z; its x-dependent part is
returned by ``manifold.log_prob_stationary_x`` and the constant log Z is
dropped (NLL on H^2 is therefore only meaningful up to a constant).
"""
import numpy as np
import scipy.special
import torch

from manifolds.hyperbolic import Hyperbolic
from manifolds.klein_bottle import KleinBottle
from manifolds.real_projective_plane import RealProjectivePlane
from manifolds.SOn import SOn
from manifolds.stiefel import Stiefel
from manifolds.torus import TorusManifold
from runner.runner_rfm import RfmRunner
from runner.runner_splitting import SplittingDiffusionRunner


def _log_volume_stiefel(n, p):
    """log Vol(St(n,p)) = sum_{i=n-p+1}^{n} log Area(S^{i-1}) under the canonical measure."""
    log_vol = 0.0
    for i in range(n - p + 1, n + 1):
        log_vol += np.log(2.0) + (i / 2.0) * np.log(np.pi) - scipy.special.gammaln(i / 2.0)
    return log_vol


def get_manifold(opt):
    md = opt.manifold_dim or {}
    name = opt.problem_name

    if name == 'torus':
        dim = int(md.get('dim', 2))
        manifold = TorusManifold(dim=dim)
        opt.nll_offset = dim * np.log(2 * np.pi)
    elif name == 'klein_bottle':
        manifold = KleinBottle()
        opt.nll_offset = 2.0 * np.log(2 * np.pi)
    elif name == 'rp2':
        manifold = RealProjectivePlane()
        opt.nll_offset = np.log(2 * np.pi)                  # Area(RP^2) = Area(S^2)/2
    elif name == 'hyperbolic':
        dim = int(md.get('dim', 2))
        potential_sigma = float(opt.potential_sigma or 1.0)
        manifold = Hyperbolic(dim=dim, potential_sigma=potential_sigma)
        opt.nll_offset = 0.0                               # log Z of the wrapped normal is dropped
    elif name == 'so3':
        manifold = SOn(3, retraction=opt.retraction)
        opt.nll_offset = np.log(8 * np.pi ** 2)
    elif name == 'qm9':
        n, p = int(md['n']), int(md['p'])
        manifold = Stiefel(n, p, retraction=opt.retraction)
        opt.nll_offset = _log_volume_stiefel(n, p)
    else:
        raise ValueError(f"Unsupported problem: {name}")

    # sigma of the velocity prior N(0, sigma^2 I) (paper Eq. (6)).  Default
    # 1/sqrt(dim M) keeps |v| ~ 1 independently of the dimension.
    if opt.velocity_sigma is None:
        opt.velocity_sigma = 1.0 / manifold.intrinsic_dim ** 0.5
        print(f'[velocity_sigma] auto-set to 1/sqrt({manifold.intrinsic_dim}) = {opt.velocity_sigma:.6f}')
    manifold.sigma = float(opt.velocity_sigma)
    return manifold


def get_dims(opt, manifold):
    """Network input / output sizes.  Splitting takes (x, v) and predicts the
    vertical score (a tangent vector); RFM takes x and predicts a tangent vector."""
    if opt.algo == 'splitting':
        return manifold.vector_dim_x + manifold.vector_dim_v, manifold.vector_dim_v
    if opt.algo == 'rfm':
        return manifold.vector_dim_x, manifold.vector_dim_v
    raise ValueError(f"Unsupported algorithm: {opt.algo}")


def get_runner(opt, manifold, net):
    if opt.algo == 'splitting':
        return SplittingDiffusionRunner(manifold=manifold, net=net, opt=opt)
    if opt.algo == 'rfm':
        return RfmRunner(manifold=manifold, net=net, opt=opt)
    raise ValueError(f"Unsupported algorithm: {opt.algo}")


_DEFAULT_DATA = {
    'torus':        './dataset/torus_gmm_t2_100000.npy',
    'klein_bottle': './dataset/klein_bottle_2d_data.npy',
    'rp2':          './dataset/rp2_gmm_data.npy',
    'hyperbolic':   './dataset/h2_wgmm.npy',
    'so3':          './dataset/rsgm_SO3_32_batch_100000.npy',
}


def get_data(opt, manifold):
    """Load the dataset as a (S, vector_dim_x) float tensor on opt.device.
    See ``generate_data/`` for how each file is produced."""
    data_path = opt.data_path or _DEFAULT_DATA.get(opt.problem_name)
    if data_path is None:
        raise ValueError(f"--data_path is required for problem {opt.problem_name}")
    data = torch.as_tensor(np.load(data_path)).to(torch.get_default_dtype())
    # torus / klein: (S, 2) angles.  rp2 / hyperbolic: (S, 3) ambient points.
    # so3: (S, 3, 3) rotation matrices.  qm9: (S, n, 4) Stiefel matrices.
    data_vec = manifold.vectorize_x(data).to(opt.device)
    print(f'[data] {data_path}: {tuple(data_vec.shape)}')
    return data_vec
