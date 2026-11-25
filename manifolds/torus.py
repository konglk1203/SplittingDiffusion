"""Flat torus T^d = [-pi, pi)^d (paper Sec. 4.1).

Points and tangent vectors are both stored as d angles.  The exponential map
is x + v (mod 2 pi) and parallel transport is the identity, so the geodesic
step of the splitting process is exact; on T^d the whole forward kernel is a
(wrapped) Gaussian, which is what makes the analytic checks of Sec. 4.1
possible (see ``analysis/splitting_target_kernel_t2.py``).
"""
import numpy as np
import torch

from .abstract_manifold import Manifold, _to_numpy

TWO_PI = 2 * np.pi


def wrap_angle(x):
    """Wrap to [-pi, pi)."""
    return (x + np.pi) % TWO_PI - np.pi


class TorusManifold(Manifold):
    def __init__(self, dim=2):
        self.dim = dim
        super().__init__(vector_dim_x=dim, vector_dim_v=dim, intrinsic_dim=dim)

    # representation: flat vector == angles
    def vectorize_x(self, x):
        return x

    def vectorize_v(self, x, v):
        return v

    def embed_x(self, x_vec):
        return x_vec

    def embed_v(self, x, v_vec):
        return v_vec

    # geometry
    def geodesic_integrator(self, x, v, alpha=1.0):
        return wrap_angle(x + v * alpha), v

    def project_to_tangent_space(self, x, v):
        return v

    def log_map(self, x, y):
        return wrap_angle(y - x)

    # probability
    def get_noise(self, x):
        return torch.randn_like(x) * self.sigma

    def sample_prior_vector(self, num_samples):
        x = (torch.rand(num_samples, self.dim) * 2 - 1) * np.pi
        v = torch.randn(num_samples, self.dim) * self.sigma
        return x, v

    def log_density_multivariate_normal(self, x, v):
        sigma2 = self.sigma ** 2
        return -0.5 * self.dim * np.log(2 * np.pi * sigma2) - 0.5 / sigma2 * torch.sum(v ** 2, dim=-1)

    # plotting
    def plot_x_vec(self, ax, x_vec, **kwargs):
        x = _to_numpy(x_vec)
        if self.dim == 1:
            ax.hist(x[:, 0], bins=60, range=(-np.pi, np.pi), color='salmon')
            return
        ax.scatter(x[:, 0], x[:, 1], s=1, color='salmon')
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)

    def plot_v_vec(self, ax, v_vec, **kwargs):
        v = _to_numpy(v_vec)
        if self.dim == 1:
            ax.hist(v[:, 0], bins=60, color='salmon')
            return
        ax.scatter(v[:, 0], v[:, 1], s=1, color='salmon')
