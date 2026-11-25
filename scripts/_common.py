"""Shared helpers for the evaluation scripts (run them from the repository root)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault('GEOMSTATS_BACKEND', 'pytorch')

import numpy as np  # noqa: E402
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import torch  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

import options  # noqa: E402
from models import get_net  # noqa: E402
from problem_factory import get_data, get_dims, get_manifold, get_runner  # noqa: E402


def load_trained_model(config, ckpt, extra_args=()):
    """Build (opt, manifold, runner) from a config + checkpoint, with the EMA
    weights copied into the network.  ``extra_args`` are forwarded to
    ``options.parse`` (e.g. ['--n_atoms', '20', '--eval_refine', '10'])."""
    opt = options.parse(['--config', config, *extra_args])
    manifold = get_manifold(opt)
    input_dim, out_dim = get_dims(opt, manifold)
    net = get_net.build(opt, input_dim, out_dim, manifold)
    runner = get_runner(opt, manifold, net)
    runner = type(runner).load_from_checkpoint(ckpt, manifold=manifold, net=net, opt=opt, map_location=opt.device)
    runner = runner.to(opt.device)
    runner.eval()
    runner._sync_ema_device()
    runner.ema.copy_to(runner.net.parameters())
    return opt, manifold, runner


def load_split(opt, manifold, split='heldout'):
    """The dataset, as a (S, dim_x) tensor: 'all', or the 'train' / 'heldout'
    part of the 90/10 split used by main.py (same seed -> same split)."""
    data = get_data(opt, manifold).cpu().numpy()
    if split == 'all':
        return torch.as_tensor(data)
    train, heldout = train_test_split(data, test_size=0.1, random_state=opt.seed)
    return torch.as_tensor(train if split == 'train' else heldout)
