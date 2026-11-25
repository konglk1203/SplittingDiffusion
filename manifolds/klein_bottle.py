"""Klein bottle K^2 (paper Sec. 4.2, Fig. 4 (a, d)): a flat, compact,
non-orientable surface.

Fundamental polygon [0, 2 pi) x [0, 2 pi) with coordinates (u, w) and the
identifications

    (u, 0) ~ (u, 2 pi)             (periodic in w)
    (0, w) ~ (2 pi, 2 pi - w)      (glued with a flip in w)

Geodesics are straight lines in the universal cover; crossing the u = 2 pi
side an odd number of times flips the w coordinate of the point *and* the w
component of the velocity (the gluing map is the reflection w -> -w).  This is
what makes the surface non-orientable, and it is the only difference from the
flat torus.
"""
import numpy as np
import torch

from .abstract_manifold import Manifold, _to_numpy

TWO_PI = 2 * np.pi


class KleinBottle(Manifold):
    def __init__(self):
        super().__init__(vector_dim_x=2, vector_dim_v=2, intrinsic_dim=2)

    @staticmethod
    def _wrap(x_raw, v=None):
        """Map a point of the universal cover R^2 back into the fundamental
        polygon; if v is given, apply the corresponding flip to it."""
        u, w = x_raw[..., 0], x_raw[..., 1]
        n_cross = torch.floor(u / TWO_PI)
        flip = (n_cross % 2 != 0)
        u = u % TWO_PI
        w = torch.where(flip, (TWO_PI - w) % TWO_PI, w % TWO_PI)
        x = torch.stack([u, w], dim=-1)
        if v is None:
            return x
        v_new = v.clone()
        v_new[..., 1] = torch.where(flip, -v[..., 1], v[..., 1])
        return x, v_new

    # representation: flat vector == (u, w)
    def vectorize_x(self, x):
        return self._wrap(x)

    def vectorize_v(self, x, v):
        return v

    def embed_x(self, x_vec):
        return x_vec

    def embed_v(self, x, v_vec):
        return v_vec

    # geometry
    def geodesic_integrator(self, x, v, alpha=1.0):
        return self._wrap(x + v * alpha, v)

    def project_to_tangent_space(self, x, v):
        return v

    def log_map(self, x, y):
        # Shortest lift ignoring the flipped gluing (adequate for nearby points).
        return (y - x + np.pi) % TWO_PI - np.pi

    # probability
    def get_noise(self, x):
        return torch.randn_like(x) * self.sigma

    def sample_prior_vector(self, num_samples):
        x = torch.rand(num_samples, 2) * TWO_PI
        v = torch.randn(num_samples, 2) * self.sigma
        return x, v

    def log_density_multivariate_normal(self, x, v):
        sigma2 = self.sigma ** 2
        return -0.5 * 2 * np.log(2 * np.pi * sigma2) - 0.5 / sigma2 * torch.sum(v ** 2, dim=-1)

    # plotting
    def plot_x_vec(self, ax, x_vec, **kwargs):
        x = _to_numpy(x_vec)
        ax.scatter(x[:, 0], x[:, 1], s=1, color='salmon', alpha=0.5)
        ax.set_xlim(0, TWO_PI)
        ax.set_ylim(0, TWO_PI)
