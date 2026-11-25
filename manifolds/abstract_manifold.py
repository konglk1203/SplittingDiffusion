"""Interface that every manifold must implement.

The splitting process (paper Sec. 3.2) only needs two geometric operations
from the manifold:

* ``get_noise``             -- a Gaussian N(0, sigma^2 I) in the tangent space
                               T_x M (the noise of the OU step, paper Eq. (8));
* ``geodesic_integrator``   -- a momentum-reversal-equivariant integrator (MREI,
                               paper Def. 2) for the geodesic step Eq. (9).

``log_map`` is only required by the RFM baseline, and ``potential_grad`` /
``log_prob_stationary_x`` only by non-compact manifolds with a confining
potential U (paper Sec. 4.2, footnote on H^2).

Representations.  Every manifold stores a point x as a flat vector of length
``vector_dim_x`` and a tangent vector v as a flat vector of length
``vector_dim_v`` (the network consumes these).  ``embed_*`` / ``vectorize_*``
convert between the flat vectors and the natural (matrix / ambient)
representation.  ``intrinsic_dim`` is d = dim M.
"""
from abc import ABC, abstractmethod

import torch


class Manifold(ABC):
    def __init__(self, vector_dim_x, vector_dim_v, intrinsic_dim):
        self.vector_dim_x = vector_dim_x
        self.vector_dim_v = vector_dim_v
        self.intrinsic_dim = intrinsic_dim
        self.sigma = 1.0            # std of the velocity prior N(0, sigma^2 I); set by problem_factory
        self.has_potential = False  # True if a confining potential U(x) != 0 is used (non-compact M)

    # ------------------------------------------------------------------ #
    #  representation                                                      #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def vectorize_x(self, x):
        """Natural representation -> flat vector of length vector_dim_x."""

    @abstractmethod
    def vectorize_v(self, x, v):
        """Natural tangent representation at x -> flat vector of length vector_dim_v."""

    @abstractmethod
    def embed_x(self, x_vec):
        """Flat vector -> natural representation."""

    @abstractmethod
    def embed_v(self, x, v_vec):
        """Flat tangent vector -> natural tangent representation at x."""

    # ------------------------------------------------------------------ #
    #  geometry                                                            #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def geodesic_integrator(self, x_vec, v_vec, alpha=1.0):
        """Momentum-reversal-equivariant integrator (paper Def. 2).

        Returns (x', v') after moving for time ``alpha`` along the geodesic
        with initial velocity v: x' = exp_x(alpha v) and v' the velocity of the
        geodesic at x' (parallel transport of v).  ``alpha`` may be negative,
        which gives the inverse map MREI^{-1} used by the reverse process.
        ``alpha`` is a float or a (B, 1) tensor.
        """

    @abstractmethod
    def project_to_tangent_space(self, x_vec, v_vec):
        """Orthogonal projection of an ambient vector onto T_x M."""

    @abstractmethod
    def log_map(self, x_vec, y_vec):
        """Riemannian logarithm log_x(y) (only needed by the RFM baseline)."""

    def tangent_inner_product(self, x_vec, u_vec, v_vec):
        """<u, v>_g at x.  Default: Euclidean inner product of the flat vectors
        (correct whenever the flat coordinates are orthonormal)."""
        return (u_vec * v_vec).sum(dim=-1)

    # ------------------------------------------------------------------ #
    #  probability                                                         #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def get_noise(self, x_vec):
        """Sample N(0, sigma^2 I) in T_x M for every x in the batch (flat vectors)."""

    @abstractmethod
    def sample_prior_vector(self, num_samples):
        """Sample (x_N, v_N) from the stationary law of the forward process
        (paper Eq. (6)): x uniform on M (or exp(-U)/Z) and v ~ N(0, sigma^2 I)."""

    @abstractmethod
    def log_density_multivariate_normal(self, x_vec, v_vec):
        """log N(v; 0, sigma^2 I) on the d-dimensional tangent space T_x M."""

    def potential_grad(self, x_vec):
        """Riemannian gradient of the confining potential U(x), as a flat
        tangent vector.  Compact manifolds use U = 0."""
        return torch.zeros((*x_vec.shape[:-1], self.vector_dim_v), device=x_vec.device, dtype=x_vec.dtype)

    def log_prob_stationary_x(self, x_vec):
        """x-dependent part of log of the stationary position density.  0 for a
        compact manifold (uniform law; the constant -log Vol(M) is handled by
        opt.nll_offset); -U(x) for a manifold with a confining potential."""
        return torch.zeros(x_vec.shape[:-1], device=x_vec.device, dtype=x_vec.dtype)

    # ------------------------------------------------------------------ #
    #  plotting (used by the validation figures)                           #
    # ------------------------------------------------------------------ #
    def plot_x_vec(self, ax, x_vec, **kwargs):
        x = _to_numpy(x_vec)
        ax.scatter(x[:, 0], x[:, 1], s=1, color='salmon')

    def plot_v_vec(self, ax, v_vec, **kwargs):
        v = _to_numpy(v_vec)
        ax.scatter(v[:, 0], v[:, 1], s=1, color='salmon')


def _to_numpy(t):
    return t.detach().cpu().numpy() if torch.is_tensor(t) else t
