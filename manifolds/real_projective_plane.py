"""Real projective plane RP^2 (paper Sec. 4.2, Fig. 4 (b, e)): compact and
non-orientable, with constant positive curvature.

RP^2 = S^2 / {x ~ -x}.  We use the closed upper hemisphere {x in S^2 : x_z >= 0}
as fundamental domain, with antipodal identification on the equator.  Since
S^2 -> RP^2 is a Riemannian double cover, exp / log / parallel transport are
the S^2 operations (from geomstats) followed by a canonicalisation step that
maps a point with x_z < 0 to -x and its velocity v to -v.
"""
import numpy as np
import torch
from geomstats.geometry.hypersphere import Hypersphere

from .abstract_manifold import Manifold, _to_numpy


class RealProjectivePlane(Manifold):
    def __init__(self):
        self.gm_sphere = Hypersphere(2)
        super().__init__(vector_dim_x=3, vector_dim_v=3, intrinsic_dim=2)

    @staticmethod
    def _canonicalize(x, v=None):
        """Fold (x, v) into the upper-hemisphere fundamental domain."""
        flip = (x[..., 2] < 0)[..., None]
        x_out = torch.where(flip, -x, x)
        if v is None:
            return x_out
        return x_out, torch.where(flip, -v, v)

    # representation: flat vector == ambient R^3 coordinates
    def vectorize_x(self, x):
        return self._canonicalize(x)

    def vectorize_v(self, x, v):
        return v

    def embed_x(self, x_vec):
        return x_vec

    def embed_v(self, x, v_vec):
        return v_vec

    # geometry
    def geodesic_integrator(self, x, v, alpha=1.0):
        if not isinstance(alpha, float):
            alpha = alpha.reshape(-1, 1)
        x_new = self.gm_sphere.metric.exp(tangent_vec=v * alpha, base_point=x)
        v_new = self.gm_sphere.metric.parallel_transport(tangent_vec=v, base_point=x, direction=v * alpha)
        x_new = torch.as_tensor(x_new).to(x.dtype)
        v_new = self.gm_sphere.to_tangent(vector=torch.as_tensor(v_new).to(v.dtype), base_point=x_new)
        return self._canonicalize(x_new, v_new)

    def project_to_tangent_space(self, x, v):
        return self.gm_sphere.to_tangent(vector=v, base_point=x)

    def log_map(self, x, y):
        # y and -y represent the same point; take the closer representative.
        same_side = (torch.sum(x * y, dim=-1, keepdim=True) >= 0)
        y_near = torch.where(same_side, y, -y)
        return torch.as_tensor(self.gm_sphere.metric.log(point=y_near, base_point=x)).to(x.dtype)

    # probability
    def get_noise(self, x):
        v = self.gm_sphere.to_tangent(vector=torch.randn_like(x), base_point=x)
        return v * self.sigma

    def sample_prior_vector(self, num_samples):
        # geomstats builds some intermediates on the CPU regardless of the
        # torch default device, so sample on the CPU and move afterwards.
        with torch.device('cpu'):
            x_sphere = self.gm_sphere.random_uniform(num_samples)
        x = self._canonicalize(torch.as_tensor(x_sphere).to(torch.get_default_dtype()).to(torch.empty(0).device))
        return x, self.get_noise(x)

    def log_density_multivariate_normal(self, x, v):
        sigma2 = self.sigma ** 2
        return -0.5 * 2 * np.log(2 * np.pi * sigma2) - 0.5 / sigma2 * torch.sum(v ** 2, dim=-1)

    # plotting: orthogonal projection of the upper hemisphere onto the unit disk
    def plot_x_vec(self, ax, x_vec, **kwargs):
        x = _to_numpy(x_vec)
        ax.scatter(x[:, 0], x[:, 1], s=1, color='salmon', alpha=0.5)
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), color='black', linewidth=0.5)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
