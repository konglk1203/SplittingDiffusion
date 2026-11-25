"""
Analytic target kernel of splitting diffusion on the flat torus T^2 with GMM
data (paper Sec. 4.1, Fig. 2 (b) and Fig. 3).

On T^2 (flat: U=0, geodesic spray S=0, exp_x(w)=(x+w) mod 2pi) every forward
splitting sub-step is an AFFINE-GAUSSIAN map of the phase-space state
z = (x, v) in R^4, so starting from a Gaussian data blob the joint law of
(x_n, v) after n split steps is EXACTLY a Gaussian (one per GMM mode, times its
periodic images). No path sampling / no DSM Monte-Carlo is needed: the whole
forward kernel, and hence splitting's regression TARGET score, is closed form.

This is the exact analytic value of splitting's training target (the h->0 /
infinite-sample limit of E[-noise/(sigxi*sigma^2) | x,v,t]). It is NOT the
continuous kinetic-Langevin kernel: it is the actual Lie-Trotter SPLIT chain's
law, which is what splitting trains against.

Convention matched to the repo (dynamics/splitting_sde.py, manifolds/torus.py):
  * fixed config: T=5, interval=20 (uniform), so h = T/interval = 0.25
  * velocity_sigma auto-set to 1/sqrt(intrinsic_dim) = 1/sqrt(2) on T^2
  * per step, forward does step_v (OU on v, x frozen) THEN step_x (transport):
        OU(h):        v <- e^{-h} v + sqrt(1-e^{-2h}) * sigma * z     (gamma=1)
                      -> on z: A_ou = blkdiag(I2, e^{-h} I2),
                         Q_ou = blkdiag(0, sigma^2 (1-e^{-2h}) I2)
        transport(h): x <- x + h v
                      -> on z: A_tr = [[I2, h I2],[0, I2]]
  * velocity prior v_0 ~ N(0, sigma^2 I);  data x_0 ~ GMM, each mode N(mu_k, C_k)
  * torus wrap period 2*pi, centered to [-pi, pi); periodic images carried as
    extra mixture components (n_images per axis).

The exact vertical score grad_v log p_n(x, v) is the mode-mixture conditional
v-score with JOINT (x,v) responsibilities:
    grad_v log p = sum_k r_k(x,v) * ( -P^{(k)}_{v|x}^{-1} (v - m^{(k)}_{v|x}) ),
    r_k(x,v) = w_k N(z; m_k, P_k) / sum_j w_j N(z; m_j, P_j).
"""
import math
import itertools
import numpy as np
import torch

TWO_PI = 2.0 * math.pi

# -------- fixed experiment config (as requested) --------
T_DEFAULT = 5.0
INTERVAL_DEFAULT = 20          # uniform grid -> h = T/interval = 0.25
GAMMA = 1.0                    # repo kinetic-Langevin friction for the tables


# ---------------------------------------------------------------------------
# Affine-Gaussian maps on z = (x1, x2, v1, v2); 2x2 blocks.
# ---------------------------------------------------------------------------
def _A_ou(h, dev, dt, g=1.0):
    """OU over effective time g*h (friction g): x unchanged, v scaled by e^{-g h}."""
    c = math.exp(-g * h)
    A = torch.eye(4, device=dev, dtype=dt)
    A[2, 2] = c
    A[3, 3] = c
    return A


def _Q_ou(h, sigma, dev, dt, g=1.0):
    """OU(g*h) additive noise covariance: v-block = sigma^2 (1-e^{-2 g h}) I."""
    q = sigma ** 2 * (1.0 - math.exp(-2.0 * g * h))
    Q = torch.zeros(4, 4, device=dev, dtype=dt)
    Q[2, 2] = q
    Q[3, 3] = q
    return Q


def _A_tr(h, dev, dt):
    """transport(h): x <- x + h v, v unchanged."""
    A = torch.eye(4, device=dev, dtype=dt)
    A[0, 2] = h
    A[1, 3] = h
    return A


def _propagate(m, P, A, Q=None):
    """Push a batch of Gaussians (m:(C,4), P:(C,4,4)) through z->A z (+N(0,Q))."""
    m2 = (A @ m.unsqueeze(-1)).squeeze(-1)
    P2 = A @ P @ A.transpose(-1, -2)
    if Q is not None:
        P2 = P2 + Q
    return m2, P2


class SplittingTargetKernelT2:
    """Exact analytic splitting target kernel on T^2 with a GMM data law.

    Parameters
    ----------
    means   : (K,2) GMM means (angles, any real; wrapped internally)
    covs    : (K,2,2) GMM covariances (data blob covariances in angle space)
    weights : (K,) mixture weights (summing to 1)
    sigma   : velocity-prior std; default 1/sqrt(2) (T^2 auto value)
    T, interval : time horizon and uniform step count (h = T/interval)
    n_images : periodic images per axis for the wrapped Gaussian (1 => 3x3=9)
    """

    def __init__(self, means, covs, weights, sigma=None, T=T_DEFAULT,
                 interval=INTERVAL_DEFAULT, n_images=1, device="cpu",
                 dtype=torch.float64, step_sizes=None, friction_gamma=1.0):
        self.device = device
        self.dtype = dtype
        self.friction_gamma = float(friction_gamma)   # velocity OU friction rate
        self.sigma = float(sigma) if sigma is not None else 1.0 / math.sqrt(2.0)
        self.T = float(T)
        self.interval = int(interval)
        # Optional non-uniform grid via `step_sizes` (length == interval). When
        # given, per-step h_n are used and ts is their cumulative sum.
        self._hs = None
        if step_sizes is not None:
            hs = [float(x) for x in step_sizes]
            assert len(hs) == self.interval, f"step_sizes len {len(hs)} != interval {self.interval}"
            self._hs = hs
            ts = [0.0]
            for hh in hs:
                ts.append(ts[-1] + hh)
            self.ts = torch.tensor(ts, device=device, dtype=dtype)
            self.T = float(ts[-1])
            self.h = None
        else:
            self.h = self.T / self.interval
            self.ts = torch.linspace(0.0, self.T, self.interval + 1, device=device, dtype=dtype)

        means = torch.as_tensor(means, device=device, dtype=dtype)       # (K,2)
        covs = torch.as_tensor(covs, device=device, dtype=dtype)         # (K,2,2)
        weights = torch.as_tensor(weights, device=device, dtype=dtype)   # (K,)
        K = means.shape[0]

        # Build wrapped initial components: one Gaussian per (mode, periodic image).
        imgs = list(itertools.product(range(-n_images, n_images + 1), repeat=2))
        comps_m, comps_P, comps_w = [], [], []
        for k in range(K):
            for (jx, jy) in imgs:
                shift = torch.tensor([jx * TWO_PI, jy * TWO_PI], device=device, dtype=dtype)
                m0 = torch.zeros(4, device=device, dtype=dtype)
                m0[:2] = means[k] + shift
                P0 = torch.zeros(4, 4, device=device, dtype=dtype)
                P0[:2, :2] = covs[k]
                P0[2, 2] = self.sigma ** 2
                P0[3, 3] = self.sigma ** 2
                comps_m.append(m0)
                comps_P.append(P0)
                comps_w.append(weights[k])
        self.w = torch.stack(comps_w)
        self.w = self.w / self.w.sum()
        m = torch.stack(comps_m)          # (C,4)
        P = torch.stack(comps_P)          # (C,4,4)
        self.C = m.shape[0]

        # Cache the POST-TRANSPORT joint Gaussian at each step n (cache index n).
        # n=0 is the initial (data x N(0,sigma^2)) law.
        self.cache_m = [m.clone()]
        self.cache_P = [P.clone()]
        g = self.friction_gamma
        for i in range(1, self.interval + 1):
            hn = self._hs[i - 1] if self._hs is not None else self.h
            A_ou = _A_ou(hn, device, dtype, g)
            Q_ou = _Q_ou(hn, self.sigma, device, dtype, g)
            A_tr = _A_tr(hn, device, dtype)
            m, P = _propagate(m, P, A_ou, Q_ou)       # OU(g*h) on v (x frozen)
            m, P = _propagate(m, P, A_tr, None)       # transport (friction-free)
            self.cache_m.append(m.clone())
            self.cache_P.append(P.clone())

    # ------------------------------------------------------------------
    def joint_gaussians(self, n, delta=0.0):
        """Per-component joint (x,v) Gaussian (mean (C,4), cov (C,4,4)) at the
        state 'after n split steps, then an extra OU(delta)'. delta in [0,h];
        delta=0 is the plain post-transport state, delta=h matches a full local
        sub-step. This is the law splitting's target is the score of."""
        m = self.cache_m[n].clone()
        P = self.cache_P[n].clone()
        if delta > 0.0:
            A_ou = _A_ou(delta, self.device, self.dtype, self.friction_gamma)
            Q_ou = _Q_ou(delta, self.sigma, self.device, self.dtype, self.friction_gamma)
            m, P = _propagate(m, P, A_ou, Q_ou)
        return m, P

    # ------------------------------------------------------------------
    @torch.no_grad()
    def vertical_score(self, x, v, n, delta=0.0):
        """Exact grad_v log p(x,v) of the analytic splitting kernel at step n
        (plus OU(delta)). x, v: (B,2). Returns (B,2).

        Uses the FULL joint (x,v) density for responsibilities (not the
        x-marginal) and the Gaussian conditional v|x for the per-mode score."""
        x = x.to(self.device, self.dtype)
        v = v.to(self.device, self.dtype)
        B = x.shape[0]
        m, P = self.joint_gaussians(n, delta)         # (C,4),(C,4,4)
        C = self.C

        # ---- fully vectorized over the C mixture components (no python loop) ----
        Pxx = P[:, :2, :2]; Pxv = P[:, :2, 2:]; Pvx = P[:, 2:, :2]; Pvv = P[:, 2:, 2:]  # (C,2,2)
        Pxx_inv = torch.linalg.inv(Pxx)                        # (C,2,2)
        Pi_inv = torch.linalg.inv(P)                           # (C,4,4)
        logdet = torch.logdet(P)                               # (C,)

        # per-component wrapped difference (B,C,4): wrap only the x-part
        dx = x[:, None, :] - m[None, :, :2]                    # (B,C,2)
        dx = (dx + math.pi) % TWO_PI - math.pi
        dv = v[:, None, :] - m[None, :, 2:]                    # (B,C,2)
        diff = torch.cat([dx, dv], dim=-1)                     # (B,C,4)
        maha = torch.einsum("bci,cij,bcj->bc", diff, Pi_inv, diff)   # (B,C)
        log_pk = -0.5 * (maha + logdet[None, :] + 4 * math.log(TWO_PI))

        # conditional v|x mean and precision, batched over C
        # cond_mean_v = m_v + Pvx Pxx^{-1} dx
        PvxPxxinv = torch.matmul(Pvx, Pxx_inv)                 # (C,2,2)
        cond_mean_v = m[None, :, 2:] + torch.einsum("cij,bcj->bci", PvxPxxinv, dx)  # (B,C,2)
        cond_cov_v = Pvv - torch.matmul(PvxPxxinv, Pxv)        # (C,2,2)
        cond_prec_v = torch.linalg.inv(cond_cov_v)             # (C,2,2)

        logr = log_pk + torch.log(self.w)[None, :]
        logr = logr - torch.logsumexp(logr, dim=1, keepdim=True)
        r = torch.exp(logr)                                    # (B,C)
        dvc = v[:, None, :] - cond_mean_v                      # (B,C,2)
        sk = -torch.einsum("cij,bcj->bci", cond_prec_v, dvc)   # (B,C,2)
        return (r[:, :, None] * sk).sum(dim=1)                 # (B,2)


# ---------------------------------------------------------------------------
# Self-test: build from a GMM params .npz and finite-difference check the score.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    PARAMS = os.environ.get("PARAMS", "dataset/torus_gmm_t2_params.npz")
    ps = dict(np.load(PARAMS))
    ker = SplittingTargetKernelT2(ps["mus"], ps["covs"], ps["weights"],
                                  device=dev, n_images=1)
    print(f"built kernel: C={ker.C} wrapped comps, sigma={ker.sigma:.4f}, "
          f"h={ker.h:.3f}, interval={ker.interval}, T={ker.T}")

    # finite-difference check of grad_v log p at a few steps/points
    def logp(x, v, n, delta):
        m, P = ker.joint_gaussians(n, delta)
        z = torch.cat([x, v], dim=-1).to(ker.dtype)
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
    for (n, delta) in [(1, 0.0), (5, ker.h), (10, 0.1), (19, ker.h)]:
        x = ((torch.rand(4000, 2, device=dev) * 2 - 1) * math.pi).to(ker.dtype)
        v = (ker.sigma * torch.randn(4000, 2, device=dev)).to(ker.dtype)
        s_ana = ker.vertical_score(x, v, n, delta)
        eps = 1e-4
        s_fd = torch.empty_like(s_ana)
        for d in range(2):
            vp = v.clone(); vp[:, d] += eps
            vm = v.clone(); vm[:, d] -= eps
            s_fd[:, d] = (logp(x, vp, n, delta) - logp(x, vm, n, delta)) / (2 * eps)
        rel = (s_ana - s_fd).norm() / s_fd.norm().clamp_min(1e-12)
        print(f"  n={n:2d} delta={delta:.3f}  rel.err(analytic vs finite-diff) = {rel:.2e}")
