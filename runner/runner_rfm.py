"""Riemannian Flow Matching baseline: training step and probability-flow NLL."""
import torch

from dynamics.rfm_ode import RiemannianFlowMatching
from losses import compute_metric_mse
from runner.runner import Runner


class RfmRunner(Runner):
    def __init__(self, manifold, net, opt):
        super().__init__(manifold, net, opt)
        self.splitting_sde = RiemannianFlowMatching(opt, manifold)

    def training_step(self, batch, batch_idx):
        """Conditional flow matching: regress the network on the geodesic
        velocity u_t of x_t = exp_{x0}(t log_{x0}(x1)), x0 ~ prior, x1 = data."""
        opt = self.opt
        x1 = batch[0].to(opt.device)
        bs = x1.shape[0]
        x0, _ = self.manifold.sample_prior_vector(bs)
        t = torch.rand(bs, device=opt.device)
        xt, ut = self.splitting_sde.sample_path(x0.to(opt.device), x1, t)
        pred = self.net(xt, None, t)
        loss = compute_metric_mse(self.manifold, xt, pred, ut)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _compute_NLL_batch(self, batch):
        """NLL via the instantaneous change of variables along the learned flow,
        integrated from the data (t=1) back to the prior (t=0) with
        opt.eval_refine Euler steps.  The divergence is estimated by a Hutchinson
        probe with a central finite difference along a geodesic, which avoids
        differentiating through the Stiefel geodesic solver."""
        opt, manifold = self.opt, self.manifold
        x = batch[0].to(opt.device)
        bs = x.shape[0]

        def vecfield(xt, t_val):
            return self.projected_net(xt, None, t_val)

        acc = torch.zeros(bs, device=opt.device)
        ts = torch.linspace(0, opt.T, int(opt.eval_refine) + 1, device=opt.device)
        for i in range(len(ts) - 1, 0, -1):
            t, dt = ts[i], ts[i] - ts[i - 1]
            x0 = x.detach()
            probe = (manifold.get_noise(x0) / manifold.sigma).detach()      # N(0, I) direction in T_x M
            eps = 1e-4
            with torch.no_grad():
                x_plus, _ = manifold.geodesic_integrator(x0, eps * probe)
                x_minus, _ = manifold.geodesic_integrator(x0, -eps * probe)
            f_plus, f_minus = vecfield(x_plus, t), vecfield(x_minus, t)
            f_val = 0.5 * (f_plus + f_minus)
            div = torch.sum(probe * (f_plus - f_minus) / (2.0 * eps), dim=-1)
            acc += div * dt
            with torch.no_grad():
                x, _ = manifold.geodesic_integrator(x, -f_val.detach() * dt, alpha=1.0)
                x = x.detach()

        logp_prior = -opt.nll_offset        # uniform prior on the compact manifold
        return -torch.mean(logp_prior - acc)
