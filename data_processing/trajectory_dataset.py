"""Datasets feeding the training loop."""
import time

import torch


class TrajectoryDataset(torch.utils.data.Dataset):
    """Cache of forward splitting chains for training (paper Algo. 3, lines 2-5).

    Holds the whole training set (S points) but simulates and serves only
    ``cache_size`` (K) chains at a time; one epoch iterates over the K cached
    chains.  Every ``traj_reuse`` epochs the runner calls
    ``resample_trajectory``, which draws K points (all of them if K = S) with
    fresh velocities v_0 ~ N(0, sigma^2 I) and re-runs Algo. 2.  Memory is
    O(K * (N+1) * dim) instead of O(S * (N+1) * dim).
    """

    def __init__(self, data, manifold, splitting_sde, cache_size=None):
        self.data = torch.as_tensor(data)                   # (S, dim_x), CPU
        self.manifold = manifold
        self.splitting_sde = splitting_sde
        S = len(self.data)
        self.cache_size = S if (cache_size is None or cache_size >= S) else int(cache_size)
        self.resample_trajectory()

    def __len__(self):
        return self.cache_size

    def __getitem__(self, idx):
        return self.x_traj[idx], self.v_traj[idx]

    @torch.no_grad()
    def resample_trajectory(self):
        t0 = time.perf_counter()
        S = len(self.data)
        x0 = self.data if self.cache_size >= S else self.data[torch.randperm(S)[:self.cache_size]]
        v0 = self.manifold.get_noise(x0.to(self.splitting_sde.opt.device)).cpu()
        self.x_traj, self.v_traj = self.splitting_sde.sample_forward_trajectory(x0, v0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[forward chains] simulated {self.cache_size}/{S} chains in {time.perf_counter() - t0:.2f}s", flush=True)


class InitialConditionDataset(torch.utils.data.Dataset):
    """Raw data points (x_0, v_0) with v_0 ~ N(0, sigma^2 I) drawn on the fly.
    Used for RFM training and for the validation set of both algorithms."""

    def __init__(self, data, manifold):
        self.data = torch.as_tensor(data)
        self.manifold = manifold

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x, self.manifold.get_noise(x)
