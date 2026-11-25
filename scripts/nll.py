"""Negative log-likelihood of a trained checkpoint on real data (paper Thm. 3
for splitting; probability-flow ODE for RFM).  Used for Tab. 2 and Tab. 3.

    python scripts/nll.py --config config/qm9/splitting.yaml --n_atoms 20 --eval_refine 10 \
        --ckpt <run>/checkpoints/snap-step=200000.ckpt --N 2048
    python scripts/nll.py --config config/qm9/rfm.yaml --n_atoms 20 --eval_refine 200 \
        --ckpt <run>/checkpoints/snap-step=200000.ckpt --N 2048

``--split heldout`` (default) evaluates on the 10% validation split of main.py;
``--split all`` draws the N points from the whole dataset.  Further flags are
forwarded to ``options.parse``.  Note (paper Sec. 4.4, Tab. 3): on St(n,4) the
value depends strongly on eval_refine and is not a reliable ranking metric.
"""
import argparse

import torch

from _common import load_split, load_trained_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--N", type=int, default=2048, help="number of data points")
    ap.add_argument("--split", choices=['heldout', 'train', 'all'], default='heldout')
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0, help="seed for the point selection and the Hutchinson probes")
    a, extra = ap.parse_known_args()

    opt, manifold, runner = load_trained_model(a.config, a.ckpt, extra)
    data = load_split(opt, manifold, a.split).to(opt.device)
    g = torch.Generator(device=data.device).manual_seed(a.seed)
    x = data[torch.randperm(data.shape[0], generator=g, device=data.device)[:a.N]]
    torch.manual_seed(a.seed)

    nll, n_done = 0.0, 0
    with torch.no_grad():
        for i in range(0, x.shape[0], a.batch):
            xb = x[i:i + a.batch]
            nll += float(runner._compute_NLL_batch((xb, None))) * xb.shape[0]
            n_done += xb.shape[0]
    nfe = opt.interval * opt.eval_refine if opt.algo == 'splitting' else opt.eval_refine
    print(f"problem={opt.problem_name} algo={opt.algo} split={a.split} N={n_done} NFE={nfe} "
          f"sigma={manifold.sigma:.4f}")
    print(f"MEAN NLL = {nll / n_done:.4f} nats")


if __name__ == "__main__":
    main()
