"""
Analytic DSM-MOMENTUM target kernel on the flat 2-torus T^2 with GMM data.

Reference kernel used in paper Fig. 2 (b, c) ("DSM-momentum"): the vertical
score of the EXACT un-split CONTINUOUS kinetic-Langevin transition kernel Eq. (5)
taken in a SINGLE step from the data endpoint (x_0, v_0 ~ N(0, sigma^2 I)) to an
absolute physical time t.  On T^2 with GMM data it is closed form.  It is only a
reference: it is available on flat manifolds only, which is the point of the
splitting construction.

Continuous kinetic Langevin (U=0, gamma=1):  dx = v dt,  dv = -v dt + sqrt(2) dW
(the diffusion matrix is Q[1,1]=2 and the transition COVARIANCE is then scaled
by sigma^2; the deterministic Phi is sigma-free). Over
absolute time t the one-step transition is Gaussian:
    [x_t - x_0 ; v_t] = Phi(t) [0 ; v_0] + N(0, sigma^2 * Sigma(t))
with (van Loan of  M = [[A, Q],[0, -A^T]],  A=[[0,1],[0,-1]], Q=[[0,0],[0,2]]):
    Phi(t) = [[1, 1-e^{-t}], [0, e^{-t}]]
    Sigma(t): s_xx = 2t-3+4e^{-t}-e^{-2t}, s_xv = (1-e^{-t})^2, s_vv = 1-e^{-2t}
(these are the sigma=1 values; multiply by sigma^2).

So, per GMM mode N(mu_k, C_k) with v_0 ~ N(0, sigma^2):
    joint (x_t, v_t) ~ N( [mu_k ; 0],  Phi diag(C_k, sigma^2 I) Phi^T + sigma^2 Sigma ).
Mixture over modes (+ periodic images on the torus). Vertical score is the
responsibility-weighted conditional v|x score, exactly as for the splitting
kernel -- the ONLY difference from splitting is the per-mode joint Gaussian:
CONTINUOUS one-shot van Loan here, vs the Lie-Trotter SPLIT chain there.

Time convention: the kernel is exposed at any absolute time t, with a per-step
accessor t_abs(n) = ts[n] (ts = linspace(0, T, interval+1)) so that it lines up
with the splitting kernel's step index n.
"""
import math
import itertools
import numpy as np
import torch

TWO_PI = 2.0 * math.pi

T_DEFAULT = 5.0
INTERVAL_DEFAULT = 20
GAMMA = 1.0


def van_loan_continuous(t, sigma):
    """Phi(t) (2x2) and sigma^2-scaled Sigma(t) (2x2) for the continuous kernel.
    Uses the closed form (gamma=1, Q[1,1]=2) to avoid any matrix-exp dependency;
    (van Loan formula for the 2x2 linear SDE)."""
    e = math.exp(-t)
    Phi = np.array([[1.0, 1.0 - e], [0.0, e]])
    s_xx = 2.0 * t - 3.0 + 4.0 * e - math.exp(-2.0 * t)
    s_xv = (1.0 - e) ** 2
    s_vv = 1.0 - math.exp(-2.0 * t)
    Sig = np.array([[s_xx, s_xv], [s_xv, s_vv]]) * (sigma ** 2)
    return Phi, Sig


class DsmMomentumKernelT2:
    """Exact analytic dsm_momentum target kernel on T^2 with a GMM data law.

    Same constructor/accessor shape as SplittingTargetKernelT2 so the two can be
    compared point-for-point. Here there is no step-by-step chain: the joint law
    at step n is the ONE-SHOT continuous kernel at absolute time t_abs(n).

    Parameters mirror the splitting kernel. `step_time` selects how step index n
    maps to absolute time:
      'right' (default): t_abs(n) = ts[n]      (so n=1..interval -> ts[1..interval],
               matching dsm_momentum's train t = train_ts[idx+1]); n=0 -> t=0 is
               the data law (handled as a tiny-time limit).
    """

    def __init__(self, means, covs, weights, sigma=None, T=T_DEFAULT,
                 interval=INTERVAL_DEFAULT, n_images=1, device="cpu",
                 dtype=torch.float64):
        self.device = device
        self.dtype = dtype
        self.sigma = float(sigma) if sigma is not None else 1.0 / math.sqrt(2.0)
        self.T = float(T)
        self.interval = int(interval)
        self.ts = torch.linspace(0.0, self.T, self.interval + 1, device=device, dtype=dtype)

        self.means = torch.as_tensor(means, device=device, dtype=dtype)      # (K,2)
        self.covs = torch.as_tensor(covs, device=device, dtype=dtype)        # (K,2,2)
        weights = torch.as_tensor(weights, device=device, dtype=dtype)
        self.K = self.means.shape[0]

        # periodic images (added as extra mixture components, like splitting)
        self.imgs = list(itertools.product(range(-n_images, n_images + 1), repeat=2))
        # replicate weights across images; normalize
        w = []
        for k in range(self.K):
            for _ in self.imgs:
                w.append(weights[k])
        self.w = torch.stack(w)
        self.w = self.w / self.w.sum()
        self.C = self.w.shape[0]

    # ------------------------------------------------------------------
    def t_abs(self, n):
        """Absolute physical time for step index n (right endpoint convention)."""
        return float(self.ts[n])

    # ------------------------------------------------------------------
    def joint_gaussians(self, t):
        """Per-component joint (x,v) Gaussian (mean (C,4), cov (C,4,4)) of the
        CONTINUOUS one-shot kernel at absolute time t. x mean is the data mode
        mean (x_0 disp mean = Phi01*0 = 0); v mean 0."""
        dev, dt = self.device, self.dtype
        t = max(float(t), 1e-6)
        Phi_np, Sig_np = van_loan_continuous(t, self.sigma)
        Phi = torch.tensor(Phi_np, device=dev, dtype=dt)
        Sig = torch.tensor(Sig_np, device=dev, dtype=dt)
        p01 = float(Phi[0, 1]); p11 = float(Phi[1, 1])

        m = torch.empty(self.C, 4, device=dev, dtype=dt)
        P = torch.empty(self.C, 4, 4, device=dev, dtype=dt)
        ci = 0
        for k in range(self.K):
            Ck = self.covs[k]                               # (2,2) data blob cov
            for (jx, jy) in self.imgs:
                shift = torch.tensor([jx * TWO_PI, jy * TWO_PI], device=dev, dtype=dt)
                mu = self.means[k] + shift
                m[ci, :2] = mu
                m[ci, 2:] = 0.0
                # per-coordinate independent kernels -> build 4x4 by coord blocks.
                # For each coord d: 2x2 joint cov of (x_d, v_d) =
                #   Phi diag(C_k[d,d], sigma^2) Phi^T + Sig   (Sig already sigma^2-scaled)
                # Off-diagonal data covariance C_k[0,1] couples x1-x2 only; it maps
                # through Phi on the x-row. We assemble the full 4x4 exactly:
                #   state order (x1,x2,v1,v2); block via Phi acting per coord.
                # Build 4x4 initial P0 = diag over coords of [[C_k, 0],[0 sigma^2]]
                P0 = torch.zeros(4, 4, device=dev, dtype=dt)
                P0[:2, :2] = Ck
                P0[2, 2] = self.sigma ** 2
                P0[3, 3] = self.sigma ** 2
                # 4x4 transition A = [[Phi00 I2, Phi01 I2],[Phi10 I2, Phi11 I2]]
                A = torch.zeros(4, 4, device=dev, dtype=dt)
                A[0, 0] = Phi[0, 0]; A[1, 1] = Phi[0, 0]
                A[0, 2] = Phi[0, 1]; A[1, 3] = Phi[0, 1]
                A[2, 0] = Phi[1, 0]; A[3, 1] = Phi[1, 0]
                A[2, 2] = Phi[1, 1]; A[3, 3] = Phi[1, 1]
                Q = torch.zeros(4, 4, device=dev, dtype=dt)
                Q[0, 0] = Sig[0, 0]; Q[1, 1] = Sig[0, 0]
                Q[0, 2] = Sig[0, 1]; Q[1, 3] = Sig[0, 1]
                Q[2, 0] = Sig[1, 0]; Q[3, 1] = Sig[1, 0]
                Q[2, 2] = Sig[1, 1]; Q[3, 3] = Sig[1, 1]
                P[ci] = A @ P0 @ A.T + Q
                ci += 1
        return m, P

    # ------------------------------------------------------------------
    @torch.no_grad()
    def vertical_score(self, x, v, t):
        """Exact grad_v log p(x,v) of the continuous one-shot kernel at absolute
        time t. x,v: (B,2). Full joint responsibilities + conditional v|x score."""
        x = x.to(self.device, self.dtype); v = v.to(self.device, self.dtype)
        B = x.shape[0]
        m, P = self.joint_gaussians(t)
        C = self.C
        log_pk = torch.empty(B, C, device=self.device, dtype=self.dtype)
        cond_mean_v = torch.empty(B, C, 2, device=self.device, dtype=self.dtype)
        cond_prec_v = torch.empty(C, 2, 2, device=self.device, dtype=self.dtype)
        for i in range(C):
            mi = m[i]; Pi = P[i]
            Pxx = Pi[:2, :2]; Pxv = Pi[:2, 2:]; Pvx = Pi[2:, :2]; Pvv = Pi[2:, 2:]
            Pxx_inv = torch.linalg.inv(Pxx)
            Pi_inv = torch.linalg.inv(Pi)
            dx = x - mi[:2]
            dx = (dx + math.pi) % TWO_PI - math.pi
            dv = v - mi[2:]
            diff = torch.cat([dx, dv], dim=-1)
            maha = torch.einsum("bi,ij,bj->b", diff, Pi_inv, diff)
            log_pk[:, i] = -0.5 * (maha + torch.logdet(Pi) + 4 * math.log(TWO_PI))
            cond_mean_v[:, i, :] = mi[2:] + (Pvx @ Pxx_inv @ dx.T).T
            cond_prec_v[i] = torch.linalg.inv(Pvv - Pvx @ Pxx_inv @ Pxv)
        logr = log_pk + torch.log(self.w)[None, :]
        logr = logr - torch.logsumexp(logr, dim=1, keepdim=True)
        r = torch.exp(logr)
        dvc = v[:, None, :] - cond_mean_v
        sk = -torch.einsum("cij,bcj->bci", cond_prec_v, dvc)
        return (r[:, :, None] * sk).sum(dim=1)


if __name__ == "__main__":
    import os
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    PARAMS = os.environ.get("PARAMS", "dataset/torus_gmm_t2_params.npz")
    ps = dict(np.load(PARAMS))
    ker = DsmMomentumKernelT2(ps["mus"], ps["covs"], ps["weights"], device=dev, n_images=1)
    print(f"built dsm_momentum kernel: C={ker.C} comps, sigma={ker.sigma:.4f}, "
          f"interval={ker.interval}, T={ker.T}")

    def logp(x, v, t):
        m, P = ker.joint_gaussians(t)
        B = x.shape[0]
        logs = torch.empty(B, ker.C, device=dev, dtype=ker.dtype)
        for i in range(ker.C):
            dx = x.to(ker.dtype) - m[i, :2]
            dx = (dx + math.pi) % TWO_PI - math.pi
            dv = v.to(ker.dtype) - m[i, 2:]
            diff = torch.cat([dx, dv], dim=-1)
            Pi_inv = torch.linalg.inv(P[i])
            maha = torch.einsum("bi,ij,bj->b", diff, Pi_inv, diff)
            logs[:, i] = -0.5 * (maha + torch.logdet(P[i]) + 4 * math.log(TWO_PI))
        return torch.logsumexp(logs + torch.log(ker.w)[None, :], dim=1)

    torch.manual_seed(0)
    for n in [1, 5, 10, 19]:
        t = ker.t_abs(n)
        x = ((torch.rand(4000, 2, device=dev) * 2 - 1) * math.pi).to(ker.dtype)
        v = (ker.sigma * torch.randn(4000, 2, device=dev)).to(ker.dtype)
        s_ana = ker.vertical_score(x, v, t)
        eps = 1e-4
        s_fd = torch.empty_like(s_ana)
        for d in range(2):
            vp = v.clone(); vp[:, d] += eps
            vm = v.clone(); vm[:, d] -= eps
            s_fd[:, d] = (logp(x, vp, t) - logp(x, vm, t)) / (2 * eps)
        rel = (s_ana - s_fd).norm() / s_fd.norm().clamp_min(1e-12)
        print(f"  n={n:2d} t={t:.3f}  rel.err(analytic vs finite-diff) = {rel:.2e}")
