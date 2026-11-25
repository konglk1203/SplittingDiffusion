"""Train a splitting-diffusion model (paper Algo. 3) or the RFM baseline.

    python main.py --config config/<problem>/<algo>.yaml [--flag value ...]

Checkpoints and the offline wandb log are written under ``<wandb_project>/<run id>/``.
Use ``--load <ckpt> --eval_only`` to only run the validation (NLL + sample plot)
of a trained model, or ``scripts/sample.py`` / ``scripts/nll.py`` for the
evaluation protocol of the paper.
"""
import logging
import math
import os
from pathlib import Path

os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
logging.getLogger("geomstats").setLevel(logging.ERROR)

import numpy as np
if not hasattr(np, "trapz"):        # numpy >= 2 removed np.trapz, which geomstats still imports
    np.trapz = np.trapezoid
import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sklearn.model_selection import train_test_split

import options
from data_processing.trajectory_dataset import InitialConditionDataset, TrajectoryDataset
from models import get_net
from problem_factory import get_data, get_dims, get_manifold, get_runner


def _yaml_safe(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _yaml_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(v) for v in value]
    return value


def log_options(wandb_logger, opt):
    """Store the fully resolved options with the run (wandb config + opt.yaml)."""
    opt_dict = {k: _yaml_safe(v) for k, v in vars(opt).items()}
    experiment = wandb_logger.experiment
    experiment.config.update(opt_dict, allow_val_change=True)
    path = Path(experiment.dir) / "opt.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(opt_dict, f, sort_keys=True)
    experiment.save(str(path), policy="now")


def main():
    opt = options.parse()
    manifold = get_manifold(opt)
    input_dim, out_dim = get_dims(opt, manifold)
    net = get_net.build(opt, input_dim, out_dim, manifold)
    print(f"device: {opt.device}  |  cuda: {torch.cuda.is_available()}  |  "
          f"params: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")
    if opt.compile:
        net = torch.compile(net)
    model = get_runner(opt, manifold, net)
    if opt.load is not None:
        print(f'Loading {opt.load}')
        model = type(model).load_from_checkpoint(opt.load, manifold=manifold, net=net, opt=opt)

    # ---- data: 90 / 10 split, the 10% held out for validation ----
    data = get_data(opt, manifold).cpu().numpy()
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=opt.seed)
    train_data, val_data = torch.tensor(train_data, device='cpu'), torch.tensor(val_data, device='cpu')

    gen = torch.Generator(device=opt.device).manual_seed(opt.seed or 0)
    if opt.algo == 'splitting':
        # splitting trains on cached forward chains (Algo. 2 / 3); RFM only needs the data points
        train_dataset = TrajectoryDataset(train_data, manifold, model.splitting_sde, cache_size=opt.cache_traj_size)
    else:
        train_dataset = InitialConditionDataset(train_data, manifold)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.train_bs_x, shuffle=True, generator=gen)
    val_loader = torch.utils.data.DataLoader(InitialConditionDataset(val_data, manifold), batch_size=opt.val_bs,
                                             generator=torch.Generator(device=opt.device))

    # ---- logging & checkpoints ----
    wandb_logger = WandbLogger(project=opt.wandb_project, group=opt.problem_name.upper(),
                               name=opt.wandb_name, log_model=True)
    log_options(wandb_logger, opt)
    validate = opt.validate or opt.eval_only
    callbacks = [ModelCheckpoint(save_last=True, save_top_k=0)]           # last.ckpt
    if validate:
        callbacks.append(ModelCheckpoint(monitor='validate/NLL_ema', mode='min', save_top_k=1,
                                         filename="best-ema-nll-{epoch}-{step}"))
    if opt.snapshot_every_n_steps > 0:
        callbacks.append(ModelCheckpoint(every_n_train_steps=opt.snapshot_every_n_steps, save_top_k=-1,
                                         filename="snap-{step}", save_on_train_epoch_end=False))

    # ---- validation cadence ----
    # Lightning only honours an integer val_check_interval if it is <= the
    # number of batches per epoch; otherwise translate the requested step
    # cadence into an epoch cadence.
    batches_per_epoch = max(1, len(train_loader))
    check_val_n_epoch, val_check_interval = opt.check_val_every_n_epoch, opt.check_val_every_n_steps
    if val_check_interval is not None:
        if val_check_interval >= batches_per_epoch:
            check_val_n_epoch = max(1, math.ceil(val_check_interval / batches_per_epoch))
            val_check_interval = None
        else:
            check_val_n_epoch = 1

    trainer = pl.Trainer(
        accelerator='gpu' if 'cuda' in opt.device else 'cpu',
        logger=wandb_logger,
        max_epochs=opt.num_epoch,
        max_steps=opt.num_iterations,
        callbacks=callbacks,
        check_val_every_n_epoch=check_val_n_epoch,
        val_check_interval=val_check_interval,
        limit_val_batches=None if validate else 0,
        inference_mode=False,          # the NLL needs autograd inside validation
        num_sanity_val_steps=0,
    )

    if opt.eval_only:
        trainer.validate(model=model, dataloaders=val_loader)
    else:
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=opt.load)


if __name__ == "__main__":
    main()
