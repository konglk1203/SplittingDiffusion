"""Shared Lightning module: optimizer / EMA / validation bookkeeping and the
sample plots.  The algorithm-specific parts (training step and NLL) live in
``runner_splitting.py`` and ``runner_rfm.py``."""
import os
from abc import abstractmethod

os.environ.setdefault('GEOMSTATS_BACKEND', 'pytorch')

import lightning.pytorch as pl
import matplotlib
import numpy as np
import torch
import transformers
import wandb
from torch_ema import ExponentialMovingAverage

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


class Runner(pl.LightningModule):
    def __init__(self, manifold, net, opt):
        super().__init__()
        self.manifold = manifold
        self.net = net
        self.opt = opt
        self.ema = ExponentialMovingAverage(net.parameters(), decay=opt.ema_decay)
        self.train_ts = opt.train_ts
        self.splitting_sde = None   # set by the subclass: the process used for sampling
        # Network output projected onto the tangent space T_x M (the score is a
        # tangent vector); used everywhere the network is evaluated at inference.
        self.projected_net = lambda x, v, t: self.manifold.project_to_tangent_space(x, net(x, v, t))

    # ----------------- EMA weights -----------------
    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema_state_dict'] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])

    def _sync_ema_device(self):
        param = next(self.net.parameters(), None)
        if param is not None:
            self.ema.to(device=param.device, dtype=param.dtype)

    def on_fit_start(self):
        self._sync_ema_device()

    def on_validation_start(self):
        self._sync_ema_device()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._sync_ema_device()
        self.ema.update()

    # ----------------- optimisation -----------------
    def configure_optimizers(self):
        opt = self.opt
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=opt.lr, weight_decay=opt.l2_norm)
        if opt.lr_schedule == 'constant':
            scheduler = transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=opt.warmup_steps)
        else:
            scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=opt.warmup_steps,
                num_training_steps=int(self.trainer.estimated_stepping_batches))
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def on_train_epoch_start(self):
        # Splitting: refresh the cached forward trajectories every N_reuse
        # epochs (paper Algo. 3, line 3).  No-op for RFM.
        R = getattr(self.opt, 'traj_reuse', None)
        ds = self.trainer.train_dataloader.dataset
        if R is not None and self.current_epoch > 0 and self.current_epoch % R == 0 \
                and hasattr(ds, 'resample_trajectory'):
            ds.resample_trajectory()
        self.net.train()

    @abstractmethod
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    @abstractmethod
    def _compute_NLL_batch(self, batch):
        raise NotImplementedError

    # ----------------- validation -----------------
    def on_validation_epoch_start(self):
        self.net.eval()

    def validation_step(self, batch, batch_idx):
        with self.ema.average_parameters():
            nll_ema = self._compute_NLL_batch(batch)
        nll = self._compute_NLL_batch(batch)
        self.log('validate/NLL', nll.detach(), on_step=False, on_epoch=True)
        self.log('validate/NLL_ema', nll_ema.detach(), on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        self.plot_backward()

    @torch.no_grad()
    def plot_backward(self):
        """Draw opt.num_plot_samples samples with the EMA weights (paper Algo. 4)
        and log snapshots of the reverse chain."""
        opt = self.opt
        self.net.eval()
        with self.ema.average_parameters():
            x, v = self.manifold.sample_prior_vector(opt.num_plot_samples)
            x_traj, v_traj = self.splitting_sde.sample_backward_trajectory(
                x.to(opt.device), v.to(opt.device), self.projected_net)
        fig = self._plot_traj(x_traj, v_traj)
        fig.suptitle(f"step {self.global_step} - reverse chain (EMA)")
        if self.logger is not None:
            if isinstance(self.logger, pl.loggers.WandbLogger):
                self.logger.experiment.log({"validate/backward_ema": wandb.Image(fig)})
            else:
                self.logger.log_image(key="validate/backward_ema", images=[fig])
        plt.close(fig)

    def _plot_traj(self, x_traj, v_traj, n_snapshot=5):
        fig, axss = plt.subplots(2, n_snapshot, figsize=(10, 4))
        total = x_traj.shape[1]
        steps = np.linspace(0, total - 1, n_snapshot).astype(int)
        for ax, k in zip(axss[0], steps):
            self.manifold.plot_x_vec(ax, x_traj[:, k])
            ax.set_title(f'x, step {k}/{total - 1}')
        for ax, k in zip(axss[1], steps):
            self.manifold.plot_v_vec(ax, v_traj[:, k])
            ax.set_title(f'v, step {k}/{total - 1}')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig
