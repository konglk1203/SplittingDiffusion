"""Stiefel manifold St(n, p) = {X in R^{n x p} : X^T X = I_p} with the canonical
metric (paper Sec. 4.4: QM9 molecules on St(n, 4)).

A point X and a tangent vector V (X^T V skew-symmetric) are stored as flat
row-major vectors of length n*p.  dim St(n,p) = np - p(p+1)/2.

Geodesic step (paper Eq. (9), MREI of Def. 2).  With A = X^T V (p x p skew)
and the thin QR  V - X A = Q R  of the horizontal part, the geodesic and its
velocity are

    [X(t), Xdot(t)] = [X, Q] exp(t L) [[I, A], [0, R]],   L = [[A, -R^T], [R, 0]]  (2p x 2p),

so only a 2p x 2p matrix exponential is needed (Edelman, Arias & Smith 1998);
the Cayley transform of L is the alternative retraction discussed in the paper
(Sec. 3.2 and App. C).  The logarithm map (Zimmermann 2017, iterative) is only
needed by the RFM baseline and is far more expensive, which is the point made
in App. B.
"""
import numpy as np
import torch

from .abstract_manifold import Manifold, _to_numpy


class Stiefel(Manifold):
    def __init__(self, n, p, retraction='exp'):
        self.n, self.p = n, p
        super().__init__(vector_dim_x=n * p, vector_dim_v=n * p, intrinsic_dim=n * p - p * (p + 1) // 2)
        assert retraction in ('exp', 'cayley')
        self.retraction = retraction

    def _exp_skew(self, S):
        """Orthogonal matrix from a batch of skew-symmetric S."""
        if self.retraction == 'exp':
            return torch.matrix_exp(S)
        I = torch.eye(S.shape[-1], device=S.device, dtype=S.dtype)      # Cayley: (I - S/2)^{-1} (I + S/2)
        return torch.linalg.solve(I - 0.5 * S, I + 0.5 * S)

    # ------------------------------------------------------------------ #
    #  representation                                                      #
    # ------------------------------------------------------------------ #
    def vectorize_x(self, X):
        if X.ndim >= 2 and X.shape[-2:] == (self.n, self.p):
            return X.reshape(*X.shape[:-2], -1)
        return X

    def vectorize_v(self, X, V):
        return self.vectorize_x(V)

    def embed_x(self, x_vec):
        if x_vec.ndim >= 2 and x_vec.shape[-2:] == (self.n, self.p):
            return x_vec
        return x_vec.reshape(*x_vec.shape[:-1], self.n, self.p)

    def embed_v(self, X, v_vec):
        return self.embed_x(v_vec)

    # ------------------------------------------------------------------ #
    #  geometry                                                            #
    # ------------------------------------------------------------------ #
    def project_to_tangent_space(self, x_vec, v_vec):
        """P_X(V) = V - X sym(X^T V)."""
        X, V = self.embed_x(x_vec), self.embed_v(None, v_vec)
        XtV = torch.matmul(X.transpose(-2, -1), V)
        V_tan = V - torch.matmul(X, 0.5 * (XtV + XtV.transpose(-2, -1)))
        return self.vectorize_v(X, V_tan)

    def project_to_manifold(self, x_vec):
        """Nearest point of St(n,p) (polar-like projection via QR with positive diagonal)."""
        Q, R = self._thin_qr(self.embed_x(x_vec))
        d = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1) + 1e-9)
        return self.vectorize_x(Q * d.unsqueeze(-2))

    def tangent_inner_product(self, x_vec, u_vec, v_vec):
        """Canonical metric <U, V> = tr(U^T V) - 1/2 tr(U^T X X^T V)."""
        X = self.embed_x(x_vec)
        U = self.embed_v(X, self.project_to_tangent_space(x_vec, u_vec))
        V = self.embed_v(X, self.project_to_tangent_space(x_vec, v_vec))
        XtU = torch.matmul(X.transpose(-2, -1), U)
        XtV = torch.matmul(X.transpose(-2, -1), V)
        return torch.sum(U * V, dim=(-2, -1)) - 0.5 * torch.sum(XtU * XtV, dim=(-2, -1))

    def geodesic_integrator(self, x_vec, v_vec, alpha=1.0):
        X = self.embed_x(x_vec)
        V = self.embed_v(X, v_vec)
        if isinstance(alpha, (float, int)):
            alpha = torch.tensor(float(alpha), device=X.device, dtype=X.dtype)
        else:
            alpha = alpha.to(device=X.device, dtype=X.dtype).reshape(-1, 1, 1)

        A = torch.matmul(X.transpose(-2, -1), V)             # p x p skew
        Q, R = self._thin_qr(V - torch.matmul(X, A))         # horizontal part
        p = self.p
        L = torch.zeros((*A.shape[:-2], 2 * p, 2 * p), device=X.device, dtype=X.dtype)
        L[..., :p, :p] = A
        L[..., :p, p:] = -R.transpose(-2, -1)
        L[..., p:, :p] = R
        E = self._exp_skew(alpha * L)
        E11, E12, E21, E22 = E[..., :p, :p], E[..., :p, p:], E[..., p:, :p], E[..., p:, p:]

        X_new = torch.matmul(X, E11) + torch.matmul(Q, E21)
        X_new = self.embed_x(self.project_to_manifold(self.vectorize_x(X_new)))   # re-orthonormalise
        # transported velocity Xdot(alpha) = [X, Q] exp(alpha L) [A; R]
        V_new = torch.matmul(X, torch.matmul(E11, A) + torch.matmul(E12, R)) \
            + torch.matmul(Q, torch.matmul(E21, A) + torch.matmul(E22, R))
        v_new = self.project_to_tangent_space(self.vectorize_x(X_new), self.vectorize_v(X_new, V_new))
        return self.vectorize_x(X_new), v_new

    def log_map(self, x_vec, y_vec, max_iter=20, tol=1e-7):
        """Riemannian logarithm log_X(Y) by the iterative algorithm of
        Zimmermann (2017).  Only used by the RFM baseline."""
        X, Y = self.embed_x(x_vec), self.embed_x(y_vec)
        p = self.p
        M = torch.matmul(X.transpose(-2, -1), Y)
        Q, N = torch.linalg.qr(Y - torch.matmul(X, M))       # vertical component
        W = torch.cat([M, N], dim=-2)                        # (2p, p)

        # Complete W to an orthogonal 2p x 2p matrix V in SO(2p).
        U, _, _ = torch.linalg.svd(W, full_matrices=True)
        V = torch.cat([W, U[..., :, p:]], dim=-1)
        det = torch.linalg.det(V)
        V = torch.cat([V[..., :, :-1], V[..., :, -1:] * det.sign().view(*det.shape, 1, 1)], dim=-1)

        # If exp(log V) != V the completion is on the wrong branch (eigenvalue
        # near -1); flipping two columns changes the branch and keeps det = 1.
        LV = self._matrix_log_batch(V)
        if torch.norm(torch.matrix_exp(LV) - V, dim=(-2, -1)).max() > 1e-3:
            V_alt = V.clone()
            V_alt[..., :, -2] *= -1
            V_alt[..., :, -1] *= -1
            LV_alt = self._matrix_log_batch(V_alt)
            err = torch.norm(torch.matrix_exp(LV) - V, dim=(-2, -1))
            err_alt = torch.norm(torch.matrix_exp(LV_alt) - V_alt, dim=(-2, -1))
            mask = (err_alt < err).view(*err.shape, 1, 1)
            V, LV = torch.where(mask, V_alt, V), torch.where(mask, LV_alt, LV)

        # Rotate the completion until the lower-right block of log V vanishes.
        for _ in range(max_iter):
            LV = self._matrix_log_batch(V)
            C = LV[..., p:, p:]
            if torch.linalg.norm(C, dim=(-2, -1)).max() < tol:
                break
            phi = torch.matrix_exp(-0.5 * (C - C.transpose(-2, -1)))
            V = torch.cat([V[..., :, :p], torch.matmul(V[..., :, p:], phi)], dim=-1)

        A, B = LV[..., :p, :p], LV[..., p:, :p]
        V_tan = torch.matmul(X, A) + torch.matmul(Q, B)
        return self.project_to_tangent_space(self.vectorize_x(X), self.vectorize_v(X, V_tan))

    @staticmethod
    def _matrix_log_batch(A):
        """Principal matrix logarithm of a batch of orthogonal matrices via
        eigendecomposition.  The non-symmetric batched ``eig`` is very slow on
        CUDA for these tiny matrices, so it runs on the CPU."""
        A_cpu = A.detach().cpu()
        vals, vecs = torch.linalg.eig(A_cpu)
        res = vecs @ torch.diag_embed(torch.log(vals)) @ torch.linalg.inv(vecs)
        return res.real.to(A.device, dtype=A.dtype)

    def _thin_qr(self, K):
        """Thin QR of a batch of n x p matrices with a positive diagonal of R.

        On the GPU this uses Cholesky-QR with one re-orthogonalisation pass
        (CholQR2), which for p <= 8 is much faster than batched Householder QR.
        Rank-deficient inputs (e.g. V ~ 0, or St(n,n) where the horizontal part
        vanishes) fall back to a fixed frame / LAPACK QR.
        """
        if K.device.type != 'cuda':
            return self._lapack_qr_posdiag(K)

        eye = torch.eye(self.p, device=K.device, dtype=K.dtype)
        G = torch.matmul(K.transpose(-2, -1), K)
        Lc, info = torch.linalg.cholesky_ex(G)
        bad = info != 0
        Lc = torch.where(bad[..., None, None], eye, Lc)
        R1 = Lc.transpose(-2, -1)
        Q1 = torch.linalg.solve_triangular(R1, K, upper=True, left=False)
        G2 = torch.matmul(Q1.transpose(-2, -1), Q1)
        Lc2, info2 = torch.linalg.cholesky_ex(G2)
        bad = bad | (info2 != 0)
        Lc2 = torch.where(bad[..., None, None], eye, Lc2)
        R2 = Lc2.transpose(-2, -1)
        Q = torch.linalg.solve_triangular(R2, Q1, upper=True, left=False)
        R = torch.matmul(R2, R1)

        if bool(bad.any()):
            idx = bad.nonzero(as_tuple=True)
            Kb = K[idx]
            amax = Kb.abs().amax(dim=(-2, -1))
            zero = amax <= 1e-6 * (1.0 + amax)
            # K == 0: R = 0 and Q drops out of every downstream product, so any
            # orthonormal frame is valid.
            Qz = torch.zeros_like(Kb)
            Qz[..., :self.p, :self.p] = eye
            Rz = torch.zeros((*Kb.shape[:-2], self.p, self.p), device=K.device, dtype=K.dtype)
            if bool((~zero).any()):
                nz = (~zero).nonzero(as_tuple=True)
                Qn, Rn = self._lapack_qr_posdiag(Kb[nz])
                Qz[nz], Rz[nz] = Qn, Rn
            Q[idx], R[idx] = Qz, Rz
        return Q, R

    @staticmethod
    def _lapack_qr_posdiag(K):
        """Exact Householder QR (on the CPU) with diag(R) forced positive."""
        Kc = K.detach().cpu() if K.device.type == 'cuda' else K
        Q, R = torch.linalg.qr(Kc)
        d = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1) + 1e-9)
        return (Q * d.unsqueeze(-2)).to(K.device, dtype=K.dtype), (R * d.unsqueeze(-1)).to(K.device, dtype=K.dtype)

    # ------------------------------------------------------------------ #
    #  probability                                                         #
    # ------------------------------------------------------------------ #
    def get_noise(self, x_vec):
        """N(0, sigma^2 I) in T_X St(n,p) under the canonical metric: V = X A + H
        with A skew (independent N(0,1) strictly-upper entries, which have
        canonical norm 1/2 ||A||_F^2) and H horizontal (X^T H = 0)."""
        X = self.embed_x(x_vec)
        batch_shape = X.shape[:-2]
        rows, cols = torch.triu_indices(self.p, self.p, offset=1, device=X.device)
        A = torch.zeros((*batch_shape, self.p, self.p), device=X.device, dtype=X.dtype)
        A[..., rows, cols] = torch.randn((*batch_shape, rows.numel()), device=X.device, dtype=X.dtype)
        A = A - A.transpose(-2, -1)
        Z = torch.randn_like(X)
        H = Z - torch.matmul(X, torch.matmul(X.transpose(-2, -1), Z))
        return self.vectorize_v(X, torch.matmul(X, A) + H) * self.sigma

    def sample_prior_vector(self, num_samples):
        """Uniform (Haar) samples of St(n,p): Q of a Gaussian n x p matrix."""
        Z = torch.randn(num_samples, self.n, self.p)
        Q, _ = self._thin_qr(Z)
        x_vec = self.vectorize_x(Q)
        return x_vec, self.get_noise(x_vec)

    def log_density_multivariate_normal(self, x_vec, v_vec):
        sigma2 = self.sigma ** 2
        v_tan = self.project_to_tangent_space(x_vec, v_vec)
        sq = self.tangent_inner_product(x_vec, v_tan, v_tan)
        return -0.5 * self.intrinsic_dim * np.log(2 * np.pi * sigma2) - 0.5 / sigma2 * sq

    # plotting: first row of X
    def plot_x_vec(self, ax, x_vec, **kwargs):
        X = _to_numpy(x_vec).reshape(-1, self.n, self.p)
        ax.scatter(X[:, 0, 0], X[:, 0, 1], s=1, color='salmon')
