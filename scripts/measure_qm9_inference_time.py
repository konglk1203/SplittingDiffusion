"""Wall-clock sampling time of trained QM9 checkpoints (paper App. B, Tab. 6:
inference time per sample at matched NFE).

    python scripts/measure_qm9_inference_time.py --n_atoms 20 --n_samples 2000 \
        --splitting_ckpt <run>/checkpoints/snap-step=200000.ckpt --splitting_eval_refine 5 \
        --rfm_ckpt <run>/checkpoints/snap-step=200000.ckpt --rfm_eval_refine 100

NFE = interval * eval_refine for splitting (20 * 5 = 100 above) and
eval_refine for RFM.  One untimed warm-up run precedes the timed repeats.
"""
import argparse
import time

import torch

from _common import load_trained_model


def time_sampling(opt, manifold, runner, n_samples):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        x0, v0 = manifold.sample_prior_vector(n_samples)
        runner.splitting_sde.sample_backward_trajectory(x0.to(opt.device), v0.to(opt.device), runner.projected_net)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_atoms", type=int, default=20)
    ap.add_argument("--n_samples", type=int, default=2000)
    ap.add_argument("--n_repeats", type=int, default=3)
    ap.add_argument("--splitting_config", default="config/qm9/splitting.yaml")
    ap.add_argument("--splitting_ckpt", required=True)
    ap.add_argument("--splitting_eval_refine", type=int, default=5)
    ap.add_argument("--rfm_config", default="config/qm9/rfm.yaml")
    ap.add_argument("--rfm_ckpt", required=True)
    ap.add_argument("--rfm_eval_refine", type=int, default=100)
    a = ap.parse_args()

    runs = {
        "splitting": (a.splitting_config, a.splitting_ckpt, a.splitting_eval_refine),
        "rfm": (a.rfm_config, a.rfm_ckpt, a.rfm_eval_refine),
    }
    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}  n_atoms={a.n_atoms}  "
          f"n_samples={a.n_samples}  n_repeats={a.n_repeats}\n")
    for algo, (config, ckpt, refine) in runs.items():
        opt, manifold, runner = load_trained_model(
            config, ckpt, ['--n_atoms', str(a.n_atoms), '--eval_refine', str(refine), '--sample_bs', str(a.n_samples)])
        nfe = opt.interval * refine if algo == 'splitting' else refine
        time_sampling(opt, manifold, runner, a.n_samples)                     # warm-up
        times = [time_sampling(opt, manifold, runner, a.n_samples) for _ in range(a.n_repeats)]
        mean_t = sum(times) / len(times)
        print(f"{algo:10s} NFE={nfe:4d}  times={[f'{t:.3f}s' for t in times]}  "
              f"per-sample={mean_t / a.n_samples * 1000:.3f} ms")


if __name__ == "__main__":
    main()
