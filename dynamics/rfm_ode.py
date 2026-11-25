"""Riemannian Flow Matching baseline (Chen & Lipman, 2023), used for the QM9
comparison of paper Sec. 4.4 and App. B-C.

Conditional path x_t = exp_{x0}(t log_{x0}(x1)) between a prior sample x0 and a
data point x1, t in [0, 1]; the regression target is the geodesic velocity.
Unlike splitting diffusion this needs the logarithm map, which on the Stiefel
manifold is an iterative solve (``Stiefel.log_map``), and one exponential map
per network evaluation at inference (paper Tab. 5).
"""
import torch
from tqdm import tqdm


class RiemannianFlowMatching:
    def __init__(self, opt, manifold):
        self.opt = opt
        self.manifold = manifold

    def _infer_ts(self):
        """Inference grid on [0, 1] with opt.eval_refine Euler steps (= NFE)."""
        n = max(1, int(getattr(self.opt, 'eval_refine', 100)))
        return torch.linspace(0.0, 1.0, n + 1, device=self.opt.device)

    def sample_path(self, x0, x1, t):
        """x_t = exp_{x0}(t log_{x0}(x1)) and the target velocity u_t (the
        transported geodesic velocity)."""
        v0 = self.manifold.log_map(x0, x1)
        return self.manifold.geodesic_integrator(x0, v0, alpha=t.reshape(-1, 1))

    @torch.no_grad()
    def sample_backward_trajectory(self, x_prior, v_dummy, net):
        """Euler integration of the learned flow from the prior (t=0) to data
        (t=1); returns (x_traj, zeros) with x_traj of shape (B, NFE+1, dim) so
        that the final sample is ``x_traj[:, -1]``."""
        opt = self.opt
        ts = self._infer_ts()
        bs = x_prior.shape[0]
        sample_bs = getattr(opt, 'sample_bs', None) or opt.train_bs_t * opt.train_bs_x
        x_traj = torch.empty((bs, len(ts), *x_prior.shape[1:]), device='cpu')

        for i in tqdm(range(0, bs, sample_bs), desc="RFM sampling", leave=False):
            end = min(i + sample_bs, bs)
            x = x_prior[i:end].to(opt.device)
            for k, t in enumerate(ts):
                x_traj[i:end, k] = x.cpu()
                if k < len(ts) - 1:
                    x, _ = self.manifold.geodesic_integrator(x, net(x, None, t), alpha=ts[k + 1] - t)
        v_traj = torch.zeros((bs, len(ts), *v_dummy.shape[1:]), device='cpu')
        return x_traj, v_traj
