"""Command-line / YAML configuration.

Every run is described by a YAML config (see ``config/``) plus optional CLI
overrides.  Precedence: CLI flag > YAML value > argparse default.  YAML files may
inherit from other YAML files through a ``base_configs`` list (later entries
override earlier ones; the file itself overrides all of its bases).

Correspondence with the notation of the paper (Sec. 3):

====================  ==========================================  =====================
paper                 meaning                                     flag
====================  ==========================================  =====================
T = t_N               terminal time                               ``--T``
N                     number of splitting intervals (Algo. 2)     ``--interval``
{t_n}, h_n            splitting scheduler (Def. 1)                ``--time_grid`` (+ ``--h_min``, ``--h_min_steps``)
gamma_n               OU friction, Eq. (8)                        ``--friction_gamma`` (constant in n)
sigma                 velocity noise scale, Eq. (5)-(6)           ``--velocity_sigma``
M_x, M_t              batch sizes in Algo. 3                      ``--train_bs_x``, ``--train_bs_t``
N_reuse               trajectory refresh period in Algo. 3        ``--traj_reuse`` (in epochs, see below)
epsilon               lower bound on the OU displacement delta    ``--delta_min_ratio`` (relative to h_n)
R_n                   reverse-OU refinement in Algo. 4            ``--eval_refine`` (constant in n)
SDE / ODE mode        reverse sampler mode in Algo. 4             ``--infer_ode`` / ``--no-infer_ode``
====================  ==========================================  =====================
"""
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROBLEMS = ['torus', 'klein_bottle', 'rp2', 'hyperbolic', 'so3', 'qm9']
ALGOS = ['splitting', 'rfm']


# --------------------------------------------------------------------------- #
#  YAML loading with inheritance                                               #
# --------------------------------------------------------------------------- #
def load_yaml_with_inheritance(config_path, _stack=None):
    """Load a YAML config, recursively merging the files listed under
    ``base_configs`` (or a single ``base_config``).  Paths are relative to the
    directory of the file that references them.  Later bases override earlier
    ones and the current file overrides all bases."""
    _stack = set() if _stack is None else set(_stack)
    config_path = Path(config_path).expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    key = str(config_path)
    if key in _stack:
        raise ValueError(f"Circular config inheritance detected at {key}")
    _stack.add(key)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}

    bases = []
    if 'base_config' in config:
        bases.append(config.pop('base_config'))
    if 'base_configs' in config:
        b = config.pop('base_configs')
        bases.extend([b] if isinstance(b, (str, Path)) else list(b))

    merged = {}
    for base in bases:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = config_path.parent / base_path
        merged.update(load_yaml_with_inheritance(base_path, _stack))
    merged.update(config)
    return merged


# --------------------------------------------------------------------------- #
#  Argument parser                                                             #
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description="Splitting diffusion on Riemannian manifolds")

    # ---------------- run / bookkeeping ----------------
    p.add_argument("--config", type=str, default=None, help="path to a YAML config (supports base_configs inheritance)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--load", type=str, default=None, help="checkpoint to load (resume training, or evaluate with --eval_only)")
    p.add_argument("--eval_only", action="store_true", help="only run validation on the loaded checkpoint")
    p.add_argument("--compile", action="store_true", help="wrap the network in torch.compile")
    p.add_argument("--wandb_project", type=str, default="SplittingDiffusion")
    p.add_argument("--wandb_name", type=str, default=None, help="run name (default: <problem>_<algo>[_N<atoms>])")
    p.add_argument("--snapshot_every_n_steps", type=int, default=1000,
                   help="save a snapshot checkpoint snap-step=<k>.ckpt every k optimizer steps (0 disables)")

    # ---------------- problem / data ----------------
    p.add_argument("--problem_name", type=str, default='torus', choices=PROBLEMS)
    p.add_argument("--manifold_dim", type=str, default=None,
                   help="manifold constructor kwargs as a YAML dict, e.g. '{dim: 2}' or '{n: 20, p: 4}'")
    p.add_argument("--data_path", type=str, default=None, help="dataset file (each problem has a default)")
    p.add_argument("--n_atoms", type=int, default=None,
                   help="[qm9] atom count n: selects dataset/qm9_by_size/qm9_N<n>.npy and St(n,4)")
    p.add_argument("--potential_sigma", type=float, default=None,
                   help="[hyperbolic] width sigma_x of the confining potential U(x)=d(x,o)^2/(2 sigma_x^2)")

    # ---------------- algorithm ----------------
    p.add_argument("--algo", type=str, default='splitting', choices=ALGOS,
                   help="'splitting' (this paper) or 'rfm' (Riemannian Flow Matching baseline)")
    p.add_argument("--retraction", type=str, default='exp', choices=['exp', 'cayley'],
                   help="[so3/qm9] MREI used for the geodesic step (paper Def. 2): exact matrix exponential or Cayley transform")

    # ---------------- splitting process (paper Sec. 3.2) ----------------
    p.add_argument("--T", type=float, default=None, help="terminal time t_N (RFM: flow horizon, always 1)")
    p.add_argument("--interval", type=int, default=20, help="N, number of splitting intervals")
    p.add_argument("--time_grid", type=str, default='uniform', choices=['uniform', 'linear'],
                   help="splitting scheduler {t_n} (Def. 1): 'uniform' h_n = T/N, or 'linear': h_min for the first "
                        "h_min_steps intervals, then step sizes grow linearly so that they sum to T (dense near the data end t=0)")
    p.add_argument("--h_min", type=float, default=None, help="[time_grid=linear] smallest step size")
    p.add_argument("--h_min_steps", type=int, default=None, help="[time_grid=linear] number of initial intervals held at h_min")
    p.add_argument("--velocity_sigma", type=float, default=None,
                   help="sigma, std of the velocity prior N(0, sigma^2 I). None = 1/sqrt(dim M), so that |v| ~ 1")
    p.add_argument("--friction_gamma", type=float, default=1.0,
                   help="gamma, OU friction rate of Eq. (8); used for every interval n")

    # ---------------- training (paper Algo. 3) ----------------
    p.add_argument("--num_iterations", type=int, default=1000000, help="total optimizer steps")
    p.add_argument("--num_epoch", type=int, default=None, help="optional cap on epochs")
    p.add_argument("--train_bs_x", type=int, default=128, help="M_x: data points per batch")
    p.add_argument("--train_bs_t", type=int, default=1, help="M_t: time samples per data point (splitting only; RFM requires 1)")
    p.add_argument("--traj_reuse", type=int, default=None,
                   help="[splitting] N_reuse: recompute the cached forward trajectories every this many epochs")
    p.add_argument("--cache_traj_size", type=int, default=None,
                   help="[splitting] number of trajectories kept in the cache (None = whole training set)")
    p.add_argument("--sample_bs", type=int, default=None,
                   help="batch size for forward/backward trajectory simulation (None = train_bs_x * train_bs_t)")
    p.add_argument("--delta_min_ratio", type=float, default=0.05,
                   help="[splitting] epsilon of Sec. 3.4: the OU displacement delta is drawn in [ratio*h_n, h_n]")
    p.add_argument("--dsm_loss_weight", type=str, default='none', choices=['local', 'none'],
                   help="[splitting] 'local': multiply the residual by sigma_xi(delta)=sqrt(1-exp(-2 gamma delta)), "
                        "i.e. the loss displayed in Algo. 3; 'none': unweighted MSE on the score (used for all "
                        "reported experiments; it up-weights small delta by 1/sigma_xi^2 and is why epsilon is needed)")
    p.add_argument("--score_parametrization", type=str, default='raw', choices=['raw', 'residual_v'],
                   help="[splitting] 'residual_v': the network output is f(x,v,t) - v, so the stationary score -v "
                        "(in the sigma^2-scaled convention) is the zero function of the network")

    # ---------------- network ----------------
    p.add_argument("--net_name", type=str, default='resnet', choices=['resnet', 'dit'])
    p.add_argument("--model_hidden_dim", type=int, default=512)
    p.add_argument("--model_blocks", type=int, default=5, help="residual blocks (resnet) / transformer blocks (dit)")
    p.add_argument("--model_num_heads", type=int, default=None, help="[dit] attention heads (default hidden_dim // 64)")

    # ---------------- optimizer ----------------
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lr_schedule", type=str, default='cosine', choices=['cosine', 'constant'],
                   help="warmup then cosine decay to 0 over num_iterations, or warmup then constant")
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--l2_norm", type=float, default=0.0, help="AdamW weight decay")
    p.add_argument("--ema_decay", type=float, default=0.9999)

    # ---------------- evaluation / sampling (paper Algo. 4) ----------------
    p.add_argument("--eval_refine", type=int, default=1,
                   help="splitting: R, reverse-OU sub-steps per interval (NFE = N*R). rfm: total number of ODE steps")
    p.add_argument("--infer_ode", action=argparse.BooleanOptionalAction, default=True,
                   help="splitting reverse sampler: probability-flow ODE (default) or SDE (--no-infer_ode)")
    p.add_argument("--validate", action=argparse.BooleanOptionalAction, default=False,
                   help="run validation (NLL + sample plots) during training. Off by default: it is slow and "
                        "the reported experiments used snapshot checkpoints instead")
    p.add_argument("--check_val_every_n_steps", type=int, default=None)
    p.add_argument("--check_val_every_n_epoch", type=int, default=None)
    p.add_argument("--val_bs", type=int, default=5000, help="validation batch size")
    p.add_argument("--num_plot_samples", type=int, default=1000, help="samples drawn for the validation plot")
    return p


def parse(argv=None):
    """Parse CLI arguments (and the YAML config they point to) into a Namespace.

    ``argv`` defaults to ``sys.argv[1:]``; scripts can pass an explicit list,
    e.g. ``options.parse(['--config', cfg, '--eval_refine', '10'])``.
    """
    parser = build_parser()
    if any(s in ('-h', '--help') for s in (sys.argv[1:] if argv is None else argv)):
        parser.parse_args(argv)                 # prints the help with defaults and exits
    defaults = {a.dest: a.default for a in parser._actions if a.dest != 'help'}
    # Parse the CLI with suppressed defaults so that we know exactly which flags
    # were given explicitly (those override the YAML).
    for a in parser._actions:
        if a.dest != 'help':
            a.default = argparse.SUPPRESS
    cli = vars(parser.parse_args(argv))

    merged = dict(defaults)
    if cli.get('config'):
        yaml_cfg = load_yaml_with_inheritance(cli['config'])
        unknown = sorted(set(yaml_cfg) - set(defaults))
        if unknown:
            raise ValueError(f"Unknown keys in {cli['config']}: {unknown}")
        merged.update(yaml_cfg)
    merged.update(cli)
    return finalize(argparse.Namespace(**merged))


# --------------------------------------------------------------------------- #
#  Derived quantities                                                          #
# --------------------------------------------------------------------------- #
def build_time_grid(opt):
    """Splitting scheduler t_0 = 0 < t_1 < ... < t_N = T (paper Def. 1).

    'uniform': h_n = T/N.
    'linear' : the first ``h_min_steps`` intervals have length ``h_min``; the
    remaining ones grow linearly from ``h_min`` to ``h_max``, with ``h_max``
    chosen so that all steps sum to T.  This puts fine steps at the data end
    (t = 0), where the final denoising happens.
    """
    N = opt.interval
    assert opt.T is not None, "--T must be set"
    if opt.time_grid == 'uniform':
        return torch.linspace(0.0, opt.T, N + 1)
    assert opt.h_min is not None and opt.h_min_steps is not None, \
        "time_grid=linear needs --h_min and --h_min_steps"
    k, M = opt.h_min_steps, N - opt.h_min_steps
    assert 1 <= k < N, f"h_min_steps ({k}) must be in [1, interval-1]"
    h_max = 2.0 * (opt.T - k * opt.h_min) / M - opt.h_min   # k*h_min + M*(h_min+h_max)/2 = T
    assert h_max >= opt.h_min, (f"derived h_max={h_max:.4g} < h_min={opt.h_min:.4g}; "
                                "increase T or decrease h_min_steps")
    hs = torch.cat([torch.full((k,), opt.h_min), torch.linspace(opt.h_min, h_max, M)])
    return torch.cumsum(torch.cat([torch.zeros(1), hs]), dim=0)


def finalize(opt):
    """Resolve derived fields (data path, time grid, device, seeds)."""
    if isinstance(opt.manifold_dim, str):
        opt.manifold_dim = yaml.safe_load(opt.manifold_dim)

    if opt.problem_name == 'qm9':
        assert opt.n_atoms is not None, "qm9 needs --n_atoms (e.g. 20 for St(20,4))"
        n = int(opt.n_atoms)
        opt.data_path = opt.data_path or f'./dataset/qm9_by_size/qm9_N{n:03d}.npy'
        opt.manifold_dim = {'n': n, 'p': 4}

    if opt.algo == 'splitting':
        assert opt.traj_reuse is not None, "splitting requires --traj_reuse (N_reuse in Algo. 3)"
    else:
        assert opt.train_bs_t == 1, "RFM draws one time per data point: set --train_bs_t 1"

    if opt.wandb_name is None:
        opt.wandb_name = f"{opt.problem_name}_{opt.algo}"
        if opt.n_atoms is not None:
            opt.wandb_name += f"_N{int(opt.n_atoms)}"

    # ---- seeds ----
    if opt.seed is not None:
        random.seed(opt.seed)
        os.environ['PYTHONHASHSEED'] = str(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        torch.backends.cudnn.benchmark = True

    # ---- device / dtype ----
    opt.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(opt.device)
    torch.set_default_dtype(torch.float32)

    # ---- time grid and per-interval friction (paper Def. 1, Eq. (8)) ----
    opt.train_ts = build_time_grid(opt)
    opt.T = float(opt.train_ts[-1])
    opt.friction_gammas = torch.full((opt.interval,), float(opt.friction_gamma))
    return opt
