"""Training step (paper Algo. 3) and NLL (paper Thm. 3) of splitting diffusion."""
import torch

from dynamics.splitting_sde import SplittingSDE
from losses import compute_metric_mse
from runner.runner import Runner


class SplittingDiffusionRunner(Runner):
    def __init__(self, manifold, net, opt):
        super().__init__(manifold, net, opt)
        self.splitting_sde = SplittingSDE(opt, manifold)

    def training_step(self, batch, batch_idx):
        """One optimizer step of Algo. 3.

        The batch holds M_x = train_bs_x cached forward chains (x_n, v_n)_{n=0..N}
        (Algo. 2).  For each of M_t = train_bs_t draws we pick an interval n and
        a displacement delta = r h_{n+1} with r ~ U[delta_min_ratio, 1], move the
        velocity along the OU leg (Eq. (10)) and regress the network on the DSM
        label of Eq. (13) at the physical time t_n + delta.
        """
        opt, net, sde, manifold = self.opt, self.net, self.splitting_sde, self.manifold
        x_traj, v_traj = batch
        ts = self.train_ts

        # time samples, shared across the M_x chains of the batch
        n_idx = torch.randint(high=opt.interval, size=(opt.train_bs_t,), device=opt.device)
        ratio = opt.delta_min_ratio + (1 - opt.delta_min_ratio) * torch.rand(opt.train_bs_t, device=opt.device)
        h = (ts[n_idx + 1] - ts[n_idx]).detach()
        delta = ratio * h                                   # OU displacement inside the interval
        t = ts[n_idx] + delta                               # physical time fed to the network

        bs_x = x_traj.shape[0]
        x = x_traj[:, n_idx].to(opt.device).reshape(-1, manifold.vector_dim_x)      # (M_x*M_t, dim_x)
        v_start = v_traj[:, n_idx].to(opt.device).reshape(-1, manifold.vector_dim_v)
        delta = delta.repeat(bs_x)
        t = t.repeat(bs_x)
        n_idx = n_idx.repeat(bs_x)

        noise = manifold.get_noise(x)                                               # N(0, sigma^2 I) in T_x M
        g = sde.friction_gammas.to(opt.device)[n_idx]
        v, label = sde.dsm_sample(x, v_start, delta, noise=noise, g=g)

        pred = manifold.project_to_tangent_space(x, net(x, v, t)).reshape(label.shape)

        # 'local' reproduces the weight sigma sqrt(1 - e^{-2 gamma delta}) of Algo. 3;
        # 'none' is the unweighted score MSE used for the reported experiments.
        reweight = sde.get_sigxi(delta, g)[..., None] if opt.dsm_loss_weight == 'local' else None
        loss = compute_metric_mse(manifold, x, pred, label, reweight=reweight)
        assert not torch.isnan(loss)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # ------------------------------------------------------------------ #
    #  NLL (paper Thm. 3)                                                  #
    # ------------------------------------------------------------------ #
    def _output_and_div(self, vecfield, x, v, probe):
        """Network output and the Hutchinson estimate probe^T (d out / d v) probe
        of its divergence in the velocity variable."""
        with torch.set_grad_enabled(True):
            v = v.detach().requires_grad_(True)
            out = vecfield(v)
            contraction = self.manifold.tangent_inner_product(x, out, probe)
            vjp = torch.autograd.grad(contraction.sum(), v, retain_graph=False, allow_unused=True)[0]
            div = torch.sum(vjp * probe, dim=-1)
        return out, div

    def _compute_NLL_batch(self, batch):
        """Negative log-likelihood of the data points via the probability-flow
        ODE of the split chain (paper Thm. 3, Eq. (14)).

        Each data point x_0 is lifted with v_0 ~ N(0, sigma^2 I).  Following the
        forward chain with the OU step replaced by its probability-flow ODE
        (``step_v`` with ode=True), the log-density changes by
        -(gamma_n d + gamma_n sigma^2 div_v s) per unit time (the geodesic step
        is volume preserving).  With score = sigma^2 s this is
        -h (gamma d + gamma div_v score).  At time T the law is the stationary one
        (Eq. (6)); the velocity marginal N(0, sigma^2 I) is divided out so that
        the result is the NLL of x_0 alone, up to the Jensen gap of using a
        single v_0 draw.
        """
        x, v = batch[0], batch[1]
        opt, manifold, net, sde = self.opt, self.manifold, self.projected_net, self.splitting_sde
        x = x.to(opt.device)
        v = manifold.get_noise(x) if v is None else v.to(opt.device)
        logp0_v = manifold.log_density_multivariate_normal(x, v)
        ts = self.train_ts
        R = int(opt.eval_refine)

        acc = 0.0
        for n in range(len(ts) - 1):
            h = (ts[n + 1] - ts[n]) / R
            g = float(sde.friction_gammas[n])
            for r in range(R):
                t_score = ts[n] + r * h
                vecfield = lambda z: net(x, z, t_score)
                probe = manifold.get_noise(x) / manifold.sigma            # N(0, I) direction
                score, div = self._output_and_div(vecfield, x, v, probe)
                x, v = sde.integrator.step_v(x, v, h, 'forward', ode=True, score=score, g=g)
                acc = acc - h * g * manifold.intrinsic_dim - h * g * div
            x, v = sde.integrator.step_x(x, v, ts[n + 1] - ts[n], 'forward')

        logpT_x = -opt.nll_offset + manifold.log_prob_stationary_x(x)   # uniform (or exp(-U)/Z) on M
        logpT_v = manifold.log_density_multivariate_normal(x, v)
        logp0_x = acc + logpT_x + logpT_v - logp0_v
        return -torch.mean(logp0_x)
