"""Splitting diffusion on the tangent bundle (paper Sec. 3.2-3.5).

The kinetic Langevin dynamics Eq. (5) on T M is split into two sub-dynamics
that are each solved exactly:

  (A) Ornstein-Uhlenbeck refresh of the velocity with x frozen, paper Eq. (8),
      whose transition kernel is the Gaussian of Eq. (10):
          v' = e^{-gamma h} v + sigma sqrt(1 - e^{-2 gamma h}) eta,   eta ~ N(0, I);
  (B) deterministic geodesic transport, paper Eq. (9), implemented by the
      manifold's momentum-reversal-equivariant integrator (MREI, Def. 2).

`SplittingIntegrator` implements the two sub-steps (and the optional
confining-potential kick for non-compact manifolds).  `SplittingSDE` runs the
forward chain (Algo. 2), produces the DSM training labels (Eq. (12)-(13)) and
runs the reverse chain (Algo. 4).

Score convention.  Everywhere in this code base the quantity called ``score``
(the network output, the DSM label, the argument of ``step_v``) is the
sigma^2-scaled vertical score

    score = sigma^2 * grad_v log p(x, v, t).

It is what the DSM label of ``dsm_sample`` estimates (label = -noise / sigma_xi
with noise ~ N(0, sigma^2 I) equals sigma^2 times the kernel score of Eq. (13)),
so the network regresses it directly; correspondingly the reverse-step
coefficients below carry no explicit sigma^2, unlike Algo. 4 in the paper.
With this convention the stationary score is simply -v.

Time convention.  The network is conditioned on the physical time t = t_n + delta
of the OU leg (paper: splitting time tau = 2 t_n + delta; the two are an
invertible relabelling of the OU legs, on which alone the score is used).
"""
import torch
from tqdm import tqdm


class SplittingIntegrator:
    """The two exactly solvable sub-steps of the splitting scheme."""

    def __init__(self, manifold):
        self.manifold = manifold

    def step_x(self, x, v, h, direction):
        """Geodesic transport by an MREI (paper Eq. (9), Def. 2).  Backward
        applies the inverse map MREI^{-1} by integrating for time -h."""
        if direction == 'backward':
            h = -h
        return self.manifold.geodesic_integrator(x, v, alpha=h)

    def step_v(self, x, v, h, direction, ode, score=None, g=1.0):
        """OU refresh of the velocity with x frozen (paper Eq. (8), (10)) and
        its time reversal (paper Eq. (11), Algo. 4).

        forward, ode=False:  v' = e^{-g h} v + sqrt(1 - e^{-2 g h}) noise,
                             noise ~ N(0, sigma^2 I)                      [Eq. (10)]
        backward, ode=False: reverse SDE with the score drift             [Algo. 4, SDE]
        backward, ode=True:  probability-flow ODE                         [Algo. 4, ODE]
        forward,  ode=True:  probability-flow ODE run forward (for the NLL of Thm. 3)

        `score` is sigma^2 grad_v log p (see module docstring); `g` is the
        friction gamma_n of the current interval.
        """
        if direction == 'forward' and not ode:
            assert score is None
        score = 0.0 if score is None else score.reshape_as(v)
        if direction == 'backward':
            h = -h
        gh = torch.as_tensor(g * h, dtype=v.dtype, device=v.device)
        decay = torch.exp(-gh)
        if ode:
            # d v = -gamma v dt - gamma sigma^2 s dt, integrated exactly in the friction term
            v = decay * v - (1 - decay) * score
        else:
            noise = self.manifold.get_noise(x)
            v = decay * v + torch.sqrt(torch.abs(1 - torch.exp(-2 * gh))) * noise - (1 - decay) * 2 * score
        return x, v

    def step_potential(self, x, v, h, direction):
        """Kick v <- v - h grad U(x) from the potential term of Eq. (9).
        No-op on compact manifolds (U = 0).  Used as a symmetric (Strang)
        half-kick around the OU and geodesic steps for non-compact manifolds."""
        if not self.manifold.has_potential:
            return x, v
        sign = 1.0 if direction == 'forward' else -1.0
        v = v - sign * h * self.manifold.potential_grad(x).reshape_as(v)
        return x, self.manifold.project_to_tangent_space(x, v)

    def forward_step(self, x, v, h, g):
        """One interval of the forward chain (Algo. 2, lines 4-5): OU refresh then geodesic transport."""
        x, v = self.step_potential(x, v, 0.5 * h, 'forward')
        x, v = self.step_v(x, v, h, 'forward', ode=False, g=g)
        x, v = self.step_x(x, v, h, 'forward')
        x, v = self.step_potential(x, v, 0.5 * h, 'forward')
        return x, v


class SplittingSDE:
    """Forward chain, DSM labels and reverse chain of splitting diffusion."""

    def __init__(self, opt, manifold):
        self.opt = opt
        self.manifold = manifold
        self.train_ts = opt.train_ts                        # t_0 = 0 < ... < t_N = T  (Def. 1)
        self.integrator = SplittingIntegrator(manifold)
        n_int = len(self.train_ts) - 1
        fg = getattr(opt, 'friction_gammas', None)          # gamma_n, one per interval (Eq. (8))
        fg = torch.ones(n_int) if fg is None else torch.as_tensor(fg, dtype=torch.get_default_dtype())
        assert fg.numel() == n_int, f"friction_gammas has {fg.numel()} entries, expected {n_int}"
        self.friction_gammas = fg

    @staticmethod
    def get_sigxi(delta, g=1.0):
        """sigma_xi(delta) = sqrt(1 - e^{-2 gamma delta}): conditional std of the OU
        kernel over a displacement delta, in units of sigma (paper Eq. (10))."""
        return torch.sqrt(torch.abs(1 - torch.exp(-2 * g * delta)))

    def _sample_batch_size(self):
        sample_bs = getattr(self.opt, 'sample_bs', None)
        if sample_bs is None:
            sample_bs = self.opt.train_bs_t * self.opt.train_bs_x
        return int(sample_bs)

    # ------------------------------------------------------------------ #
    #  DSM label (paper Sec. 3.4)                                          #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def dsm_sample(self, x, v_start, delta, noise, g=1.0):
        """Draw v ~ q_delta(. | v_start) along the OU leg starting at the grid
        point (x_n, v_n) = (x, v_start) and return (v, label) with

            v     = e^{-g delta} v_start + sigma_xi(delta) noise,      noise ~ N(0, sigma^2 I)
            label = -noise / sigma_xi(delta) = sigma^2 grad_v log q_delta(v | v_start)   [Eq. (13)]

        By the DSM identity Eq. (12), E[label | x, v] = sigma^2 grad_v log p_t(x, v),
        so regressing the network on `label` learns the (sigma^2-scaled) score.
        """
        delta = delta[:, None]
        if torch.is_tensor(g):
            g = g.reshape(delta.shape[0], *([1] * (delta.dim() - 1)))
        sigxi = self.get_sigxi(delta, g)
        v = torch.exp(-g * delta) * v_start + sigxi * noise
        return v, -noise / sigxi

    # ------------------------------------------------------------------ #
    #  Forward chain (paper Algo. 2)                                       #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample_forward_trajectory(self, x0, v0):
        """Run Algo. 2 from (x_0, v_0) and return the whole chain
        (x_n, v_n)_{n=0..N} as two CPU tensors of shape (B, N+1, dim)."""
        opt, ts = self.opt, self.train_ts
        bs = x0.shape[0]
        sample_bs = self._sample_batch_size()
        x_traj = torch.empty((bs, len(ts), *x0.shape[1:]), device='cpu')
        v_traj = torch.empty((bs, len(ts), *v0.shape[1:]), device='cpu')

        for i in tqdm(range(0, bs, sample_bs), desc="forward chain", leave=False):
            end = min(i + sample_bs, bs)
            x, v = x0[i:end].to(opt.device), v0[i:end].to(opt.device)
            x_traj[i:end, 0], v_traj[i:end, 0] = x.cpu(), v.cpu()
            for n in range(1, len(ts)):
                h = ts[n] - ts[n - 1]
                x, v = self.integrator.forward_step(x, v, h, g=float(self.friction_gammas[n - 1]))
                x_traj[i:end, n], v_traj[i:end, n] = x.cpu(), v.cpu()
        return x_traj, v_traj

    # ------------------------------------------------------------------ #
    #  Reverse chain (paper Algo. 4)                                       #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample_backward_trajectory(self, xN, vN, net):
        """Run Algo. 4 from (x_N, v_N) ~ stationary law with the score network
        `net(x, v, t)`; returns the chain (x_n, v_n)_{n=0..N} on the CPU.

        Per interval n = N..1: geodesic step MREI^{-1}(x_n, v_n, h_n), then
        R = opt.eval_refine reverse-OU sub-steps of length h_n / R (ODE if
        opt.infer_ode else SDE), the network being evaluated at the physical
        time of the end of each sub-step.  NFE = N * R.
        """
        opt, ts = self.opt, self.train_ts
        bs = xN.shape[0]
        sample_bs = self._sample_batch_size()
        R = max(1, int(getattr(opt, 'eval_refine', 1)))
        use_ode = bool(getattr(opt, 'infer_ode', True))
        x_traj = torch.empty((bs, len(ts), *xN.shape[1:]), device='cpu')
        v_traj = torch.empty((bs, len(ts), *vN.shape[1:]), device='cpu')

        for i in tqdm(range(0, bs, sample_bs), desc="reverse chain", leave=False):
            end = min(i + sample_bs, bs)
            x, v = xN[i:end].to(opt.device), vN[i:end].to(opt.device)
            x_traj[i:end, -1], v_traj[i:end, -1] = x.cpu(), v.cpu()
            for n in range(len(ts) - 1, 0, -1):
                h = ts[n] - ts[n - 1]
                g = float(self.friction_gammas[n - 1])
                x, v = self.integrator.step_potential(x, v, 0.5 * h, 'backward')
                x, v = self.integrator.step_x(x, v, h, 'backward')
                for r in range(1, R + 1):
                    t_sub = ts[n] - r * (h / R)
                    score = net(x, v, t_sub)
                    x, v = self.integrator.step_v(x, v, h / R, 'backward', use_ode, score, g=g)
                x, v = self.integrator.step_potential(x, v, 0.5 * h, 'backward')
                x_traj[i:end, n - 1], v_traj[i:end, n - 1] = x.cpu(), v.cpu()
        return x_traj, v_traj
