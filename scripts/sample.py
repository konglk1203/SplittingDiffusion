"""Draw samples from a trained checkpoint (paper Algo. 4 for splitting; the
learned flow for RFM) and save them as a .npy array of points on the manifold.

    python scripts/sample.py --config config/klein_bottle/splitting.yaml --ckpt <run>/checkpoints/last.ckpt \
        --out samples/klein.npy --N 3000 --no-infer_ode
    python scripts/sample.py --config config/qm9/splitting.yaml --n_atoms 20 --eval_refine 10 \
        --ckpt <run>/checkpoints/snap-step=200000.ckpt --out samples/sp20.npy --N 5000
    python scripts/sample.py --config config/qm9/rfm.yaml --n_atoms 20 --eval_refine 200 \
        --ckpt <run>/checkpoints/snap-step=200000.ckpt --out samples/rfm20.npy --N 5000

Any further flag (``--n_atoms``, ``--eval_refine``, ``--no-infer_ode``, ...) is
forwarded to ``options.parse`` and overrides the config.  Splitting uses
NFE = interval * eval_refine; RFM uses NFE = eval_refine.
"""
import argparse
import os

import numpy as np
import torch

from _common import load_trained_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--N", type=int, default=5000, help="number of samples")
    ap.add_argument("--sample_bs", type=int, default=20000, help="samples simulated per batch")
    a, extra = ap.parse_known_args()

    opt, manifold, runner = load_trained_model(a.config, a.ckpt, extra + ['--sample_bs', str(a.sample_bs)])
    x0, v0 = manifold.sample_prior_vector(a.N)
    with torch.no_grad():
        x_traj, _ = runner.splitting_sde.sample_backward_trajectory(x0.to(opt.device), v0.to(opt.device),
                                                                   runner.projected_net)
    # splitting runs from t_N (index -1) down to t_0 (index 0); RFM from t=0 (prior) to t=1 (data)
    gen = x_traj[:, 0] if opt.algo == 'splitting' else x_traj[:, -1]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.save(a.out, gen.numpy())
    nfe = opt.interval * opt.eval_refine if opt.algo == 'splitting' else opt.eval_refine
    print(f"saved {a.out} {tuple(gen.shape)}  (problem={opt.problem_name}, algo={opt.algo}, NFE={nfe}, "
          f"sampler={'ODE' if opt.algo == 'rfm' or opt.infer_ode else 'SDE'})")


if __name__ == "__main__":
    main()
