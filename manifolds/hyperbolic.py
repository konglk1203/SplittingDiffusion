"""Hyperbolic space H^n in the Lorentz (hyperboloid) model (paper Sec. 4.2,
Fig. 4 (c, f)): the non-compact example.

H^n = {x in R^{n+1} : <x, x>_L = -1, x_0 > 0},  <x, y>_L = -x_0 y_0 + sum_i x_i y_i.
T_x H^n = {v : <x, v>_L = 0}, on which <., .>_L is positive definite.

Because H^n has infinite volume there is no uniform stationary law, so the
kinetic Langevin dynamics Eq. (5) is run with the confining potential

    U(x) = d_H(x, o)^2 / (2 sigma_x^2),       grad U(x) = -log_x(o) / sigma_x^2,

whose stationary position law is the wrapped normal centred at the origin o
(paper Sec. 4.2, footnote).  The splitting integrator applies -grad U as a
symmetric half-kick around the OU / geodesic steps (``SplittingIntegrator``).

Points and tangent vectors are stored as ambient (n+1)-vectors; intrinsic_dim = n.
"""
import numpy as np
import torch

from .abstract_manifold import Manifold, _to_numpy

_EPS = 1e-7
_MAX_STEP = 30.0   # clamp |v| so that cosh / sinh never overflow (never active on a sane grid)


def _mink_dot(x, y):
    """Minkowski inner product, keeping the last axis."""
    prod = x * y
    return (prod[..., 1:].sum(dim=-1) - prod[..., 0])[..., None]


class Hyperbolic(Manifold):
    def __init__(self, dim=2, potential_sigma=1.0):
        self.dim = dim
        super().__init__(vector_dim_x=dim + 1, vector_dim_v=dim + 1, intrinsic_dim=dim)
        self.has_potential = True
        self.potential_sigma = float(potential_sigma)

    # representation: flat vector == ambient Lorentz coordinates
    def vectorize_x(self, x):
        return x

    def vectorize_v(self, x, v):
        return v

    def embed_x(self, x_vec):
        return x_vec

    def embed_v(self, x, v_vec):
        return v_vec

    # ---------------- Lorentz geometry ----------------
    @staticmethod
    def _project_to_hyperboloid(x):
        spatial = x[..., 1:]
        x0 = torch.sqrt(1.0 + (spatial ** 2).sum(dim=-1, keepdim=True))
        return torch.cat([x0, spatial], dim=-1)

    def project_to_tangent_space(self, x, v):
        return v + _mink_dot(x, v) * x

    @staticmethod
    def _tangent_norm(v):
        return torch.sqrt(_mink_dot(v, v).clamp_min(0.0) + _EPS)

    def tangent_inner_product(self, x, u, v):
        return _mink_dot(self.project_to_tangent_space(x, u), self.project_to_tangent_space(x, v)).squeeze(-1)

    def exp_map(self, x, v):
        """exp_x(v) = cosh|v| x + sinh|v| v/|v|."""
        v = self.project_to_tangent_space(x, v)
        vn = self._tangent_norm(v)
        vn_c = vn.clamp(max=_MAX_STEP)
        v = v * (vn_c / vn)
        y = torch.cosh(vn_c) * x + (torch.sinh(vn_c) / vn_c) * v
        return self._project_to_hyperboloid(y)

    def log_map(self, x, y):
        """log_x(y) = d(x,y) u/|u| with u = y + <x,y>_L x."""
        xy = _mink_dot(x, y).clamp_max(-1.0 - _EPS)
        d = torch.acosh(-xy)
        u = y + xy * x
        return self.project_to_tangent_space(x, d * u / self._tangent_norm(u))

    def _parallel_transport(self, x, y, v):
        """Transport v in T_x along the geodesic to T_y (closed form)."""
        xy = _mink_dot(x, y)
        w = v + (_mink_dot(y, v) / (1.0 - xy)) * (x + y)
        return self.project_to_tangent_space(y, w)

    def geodesic_integrator(self, x, v, alpha=1.0):
        v = self.project_to_tangent_space(x, v)
        if not isinstance(alpha, float):
            alpha = alpha.reshape(-1, 1)
        x_new = self.exp_map(x, v * alpha)
        v_new = self._parallel_transport(x, x_new, v)
        return x_new, v_new

    def origin(self, like=None, num=None):
        """The base point o = (1, 0, ..., 0)."""
        if like is not None:
            o = torch.zeros_like(like)
        else:
            o = torch.zeros((num, self.vector_dim_x), dtype=torch.get_default_dtype())
        o[..., 0] = 1.0
        return o

    def _dist_to_origin(self, x):
        return torch.acosh(-_mink_dot(x, self.origin(like=x)).clamp_max(-1.0 - _EPS))

    # ---------------- probability ----------------
    def get_noise(self, x):
        """N(0, sigma^2 I) in T_x: draw a standard normal in the orthonormal
        frame of T_o (= the spatial coordinates) and parallel-transport it to x
        (an isometry).  Projecting an ambient normal onto T_x would *not* give
        the right scale away from the origin."""
        o = self.origin(like=x)
        z = torch.zeros_like(x)
        z[..., 1:] = torch.randn((*x.shape[:-1], self.dim), device=x.device, dtype=x.dtype)
        return self._parallel_transport(o, x, z) * self.sigma

    def sample_prior_vector(self, num_samples):
        """Stationary law: x ~ wrapped normal exp_o(sigma_x N(0, I)), v ~ N(0, sigma^2 I)."""
        o = self.origin(num=num_samples)
        tang = torch.zeros_like(o)
        tang[..., 1:] = torch.randn((num_samples, self.dim), dtype=o.dtype)
        x = self.exp_map(o, self.potential_sigma * tang)
        return x, self.get_noise(x)

    def log_density_multivariate_normal(self, x, v):
        sigma2 = self.sigma ** 2
        v = self.project_to_tangent_space(x, v)
        sq = _mink_dot(v, v).squeeze(-1).clamp_min(0.0)
        return -0.5 * self.dim * np.log(2 * np.pi * sigma2) - 0.5 / sigma2 * sq

    # ---------------- confining potential ----------------
    def potential_grad(self, x):
        """grad U = -log_x(o) / sigma_x^2."""
        return -self.log_map(x, self.origin(like=x)) / self.potential_sigma ** 2

    def log_prob_stationary_x(self, x):
        """-U(x) (the normalising constant is dropped)."""
        return -0.5 * self._dist_to_origin(x).squeeze(-1) ** 2 / self.potential_sigma ** 2

    # ---------------- plotting: Poincare disk ----------------
    def plot_x_vec(self, ax, x_vec, **kwargs):
        x = _to_numpy(x_vec)
        u, w = x[:, 1] / (1 + x[:, 0]), x[:, 2] / (1 + x[:, 0])
        ax.scatter(u, w, s=1, color='salmon', alpha=0.5)
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), color='black', linewidth=0.5)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect('equal')
