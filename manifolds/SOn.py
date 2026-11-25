"""Special orthogonal group SO(n) (paper Sec. 4.3, n = 3).

A point is an n x n rotation matrix g (stored flat, n^2 numbers).  A tangent
vector at g is the left-trivialised Lie-algebra element xi in so(n), stored as
its n(n-1)/2 strictly-upper-triangular entries (the velocity is v = g xi).  In
these coordinates the geodesic step Eq. (9) is g <- g exp(alpha xi) with xi
unchanged (left-invariant metric: parallel transport along a geodesic is the
identity on the trivialised velocity), so the MREI of Def. 2 is the matrix
exponential (exact) or, optionally, the Cayley transform.
"""
import numpy as np
import torch

from .abstract_manifold import Manifold, _to_numpy


class SOn(Manifold):
    def __init__(self, dim, retraction='exp'):
        self.dim = dim
        vector_dim_v = dim * (dim - 1) // 2
        super().__init__(vector_dim_x=dim ** 2, vector_dim_v=vector_dim_v, intrinsic_dim=vector_dim_v)
        assert retraction in ('exp', 'cayley')
        self.retraction = retraction
        self._triu = torch.triu_indices(dim, dim, offset=1)

    def _exp_skew(self, S):
        """Orthogonal matrix from a batch of skew-symmetric S."""
        if self.retraction == 'exp':
            return torch.matrix_exp(S)
        I = torch.eye(S.shape[-1], device=S.device, dtype=S.dtype)      # Cayley: (I - S/2)^{-1} (I + S/2)
        return torch.linalg.solve(I - 0.5 * S, I + 0.5 * S)

    # representation
    def vectorize_x(self, g):
        return g.reshape(g.shape[0], -1)

    def vectorize_v(self, g, xi):
        idx = self._triu.to(xi.device)
        return xi[:, idx[0], idx[1]]

    def embed_x(self, g_vec):
        return g_vec.reshape(g_vec.shape[0], self.dim, self.dim)

    def embed_v(self, g, xi_vec):
        idx = self._triu.to(xi_vec.device)
        skew = torch.zeros(xi_vec.shape[0], self.dim, self.dim, dtype=xi_vec.dtype, device=xi_vec.device)
        skew[:, idx[0], idx[1]] = xi_vec
        return skew - skew.transpose(-2, -1)

    # geometry
    def geodesic_integrator(self, g_vec, xi_vec, alpha=1.0):
        g = self.embed_x(g_vec)
        xi = self.embed_v(g, xi_vec)
        if not isinstance(alpha, float):
            alpha = alpha.reshape(-1, 1, 1)
        g_new = torch.matmul(g, self._exp_skew(xi * alpha))
        return self.vectorize_x(g_new), xi_vec

    def project_to_tangent_space(self, g_vec, xi_vec):
        return xi_vec

    def log_map(self, g_vec, h_vec):
        """log_g(h) = log(g^T h) in so(n) (principal branch), via the
        eigendecomposition of the rotation g^T h.  Only the RFM baseline needs
        this; the splitting process never uses a logarithm."""
        g, h = self.embed_x(g_vec), self.embed_x(h_vec)
        rel = torch.matmul(g.transpose(-2, -1), h).to(torch.complex64)
        vals, vecs = torch.linalg.eig(rel)            # eigenvalues on the unit circle
        A = vecs @ torch.diag_embed(torch.log(vals)) @ torch.linalg.inv(vecs)
        A = 0.5 * (A - A.transpose(-2, -1)).real.to(g.dtype)
        return self.vectorize_v(g, A)

    # probability
    def get_noise(self, g_vec):
        return torch.randn(*g_vec.shape[:-1], self.vector_dim_v, device=g_vec.device, dtype=g_vec.dtype) * self.sigma

    def sample_haar(self, batch):
        """Uniform (Haar) samples of SO(n): QR of a Gaussian matrix, with the
        phase of R fixed and det corrected to +1."""
        A = torch.randn(batch, self.dim, self.dim)
        Q, R = torch.linalg.qr(A)
        d = torch.diagonal(R, dim1=-2, dim2=-1)
        Q = Q * (d / d.abs()).unsqueeze(-2)
        neg = torch.linalg.det(Q) < 0
        Q[neg, :, 0] *= -1
        return Q

    def sample_prior_vector(self, num_samples):
        g = self.sample_haar(num_samples)
        xi = torch.randn(num_samples, self.vector_dim_v) * self.sigma
        return self.vectorize_x(g), xi

    def log_density_multivariate_normal(self, g_vec, xi_vec):
        sigma2 = self.sigma ** 2
        return -0.5 * self.vector_dim_v * np.log(2 * np.pi * sigma2) - 0.5 / sigma2 * torch.sum(xi_vec ** 2, dim=-1)

    # plotting: two entries of the rotation matrix
    def plot_x_vec(self, ax, g_vec, **kwargs):
        g = _to_numpy(g_vec).reshape(-1, self.dim, self.dim)
        ax.scatter(g[:, 0, 0], g[:, 0, 2], s=1, color='salmon')

    def plot_v_vec(self, ax, xi_vec, **kwargs):
        xi = _to_numpy(xi_vec)
        ax.scatter(xi[:, 0], xi[:, 2 if xi.shape[1] > 2 else 1], s=1, color='salmon')
