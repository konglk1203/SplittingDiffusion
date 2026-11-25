# Splitting Diffusion on Riemannian Manifolds

Code for **"Enabling Denoising Score Matching Type Training for Manifold Diffusion
Model via Momentum and Splitting"** (Kong, Zhu, Tao).

Splitting diffusion lifts the diffusion to the tangent bundle and replaces the
intractable Brownian forward process by a splitting scheme that alternates

1. an Ornstein-Uhlenbeck update of the velocity in the (Euclidean) tangent space,
   whose transition kernel is a Gaussian in closed form, and
2. a deterministic geodesic step on the manifold.

The score is therefore learned by denoising score matching in the tangent fibre,
and the only geometric oracle needed is the exponential map (or any
momentum-reversal-equivariant retraction).  The repository contains the method,
the Riemannian Flow Matching baseline used in the paper, six manifolds and
everything needed to reproduce the figures and tables.

## Installation

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126   # pick your CUDA version
pip install -r requirements.txt
export WANDB_MODE=offline                                              # logging is local by default
```

Python >= 3.10.  Everything runs on a single GPU (a V100 was used for the paper);
the low-dimensional examples also run on a CPU.

## Repository layout

```
main.py                     training entry point (paper Algo. 3)
options.py                  all options, YAML inheritance, time grid (Def. 1)
problem_factory.py          problem name -> manifold, dataset, runner
dynamics/splitting_sde.py   the splitting process: OU step, geodesic step, forward chain (Algo. 2),
                            DSM labels (Eq. 12-13), reverse chain (Algo. 4)
dynamics/rfm_ode.py         Riemannian Flow Matching baseline
runner/                     Lightning modules: training step, NLL (Thm. 3), validation plots
manifolds/                  T^d, Klein bottle, RP^2, H^2, SO(n), St(n,p)
models/                     ResNet score network, DiT for St(n,p)
config/                     one folder per experiment (+ config/_base shared pieces)
generate_data/              synthetic datasets and the QM9 preprocessing
scripts/                    sampling, NLL, QM9 energy distance, timing
analysis/                   analytic T^2 kernels and the notebook that draws Fig. 2-4
docs/qm9_results.md         evaluation protocol and raw numbers of Sec. 4.4 / App. B
```

## Quick start

```bash
python generate_data/generate_torus_gmm.py                 # dataset/torus_gmm_t2.npy (+ mixture parameters)
python main.py --config config/torus/splitting.yaml        # ~4000 steps, a few minutes on a GPU
python scripts/sample.py --config config/torus/splitting.yaml \
    --ckpt SplittingDiffusion/<run id>/checkpoints/last.ckpt --out samples/torus.npy --N 5000
```

Checkpoints are written to `SplittingDiffusion/<wandb run id>/checkpoints/`
(`last.ckpt` and a snapshot every `--snapshot_every_n_steps`); the fully
resolved options of a run are saved next to its wandb log as `opt.yaml`.
Any option can be overridden on the command line, e.g.
`--velocity_sigma 0.5 --eval_refine 10 --no-infer_ode`.

## Where the paper lives in the code

| Paper | Code |
|---|---|
| Eq. (5)-(6): kinetic Langevin on TM, stationary law `exp(-U) N(0, sigma^2 I)` | `manifolds/*.py::sample_prior_vector`, `get_noise`; `--velocity_sigma` is sigma |
| Eq. (8), (10): OU velocity step, closed-form kernel | `SplittingIntegrator.step_v` (forward, `ode=False`) |
| Eq. (9), Def. 2: geodesic step by an MREI | `SplittingIntegrator.step_x` -> `Manifold.geodesic_integrator` (exact exp map; `--retraction cayley` on SO(n)/St(n,p)); the `-grad U` term is `step_potential` (H^2 only) |
| Def. 1: splitting scheduler `{t_n}` | `options.build_time_grid`; `config/_base/grid_uniform.yaml`, `grid_linear.yaml`; `--interval` is N |
| Algo. 2: forward chain | `SplittingSDE.sample_forward_trajectory`, cached by `TrajectoryDataset` |
| Eq. (11): reverse OU | `SplittingIntegrator.step_v` (backward) |
| Eq. (12)-(13), Sec. 3.4: DSM label and loss | `SplittingSDE.dsm_sample`, `SplittingDiffusionRunner.training_step`, `losses.compute_metric_mse` |
| Remark 1: epsilon away from the interval endpoint | `--delta_min_ratio` (delta is drawn in `[0.05 h_n, h_n]`) |
| Algo. 3: training | `main.py`, `runner/runner_splitting.py`; `M_x = --train_bs_x`, `M_t = --train_bs_t`, `N_reuse = --traj_reuse` (in epochs) |
| Algo. 4: inference, refinement `R_n`, SDE/ODE | `SplittingSDE.sample_backward_trajectory`; `--eval_refine` is R, `--infer_ode` / `--no-infer_ode` |
| Thm. 3, Eq. (14): NLL | `SplittingDiffusionRunner._compute_NLL_batch`, `scripts/nll.py` |
| Sec. 4.1, Fig. 2-3: T^2 checks | `analysis/demo_splitting.ipynb` Parts 1-2, `analysis/splitting_target_kernel_t2.py`, `analysis/dsm_momentum_kernel_t2.py` |
| Sec. 4.2, Fig. 4: Klein bottle, RP^2, H^2 | `manifolds/klein_bottle.py`, `real_projective_plane.py`, `hyperbolic.py`; `config/{klein_bottle,rp2,hyperbolic}/`; notebook Parts 3-4 |
| Sec. 4.3, Tab. 2: SO(3) | `manifolds/SOn.py`, `config/so3/`, `scripts/nll.py` |
| Sec. 4.4, Tab. 3-4: QM9 on St(n,4) | `manifolds/stiefel.py`, `models/dit.py`, `config/qm9/`, `generate_data/preprocess_qm9_by_size.py`, `scripts/{sample,nll,qm9_sample_metric}.py` |
| App. B, Tab. 5-6: cost | `scripts/measure_qm9_inference_time.py`; training times in `docs/qm9_results.md` |
| App. C, Tab. 7: hyperparameters; energy distance | `config/qm9/_problem.yaml`, `splitting.yaml`, `rfm.yaml`; `scripts/qm9_sample_metric.py` |
| RFM baseline | `dynamics/rfm_ode.py`, `runner/runner_rfm.py`, `Manifold.log_map` |

## Reproducing the experiments

### Sec. 4.1 - correctness checks on T^2 (Fig. 2, Fig. 3)

No training is involved: on the flat torus the forward kernel of the split
chain is a Gaussian mixture in closed form, so the analytic score, the
single-sample label variance and the DSM identity can be evaluated exactly.

```bash
python generate_data/generate_torus_gmm.py              # mixture + its parameters (needed by the analytic kernel)
jupyter notebook analysis/demo_splitting.ipynb          # Parts 1-2 write figures/sp_t2_*.pdf and figures/gmm_t2_*.pdf
```

`analysis/build_notebook.py` regenerates the notebook from source.

### Sec. 4.2 - Klein bottle, RP^2, H^2 (Fig. 4)

```bash
python generate_data/generate_klein_bottle_data.py
python generate_data/generate_rp2_data.py
python generate_data/generate_h2_data.py

python main.py --config config/klein_bottle/splitting.yaml      # 6000 steps
python main.py --config config/rp2/splitting.yaml               # 20000 steps
python main.py --config config/hyperbolic/splitting.yaml        # 3000 steps

python scripts/sample.py --config config/klein_bottle/splitting.yaml --ckpt <ckpt> --out samples/klein.npy --N 3000
```

The three configs use the SDE sampler (`infer_ode: false`).  Parts 3-4 of the
notebook run exactly these commands and draw the data / generated panels.

### Sec. 4.3 - Gaussian mixture on SO(3) (Tab. 2)

```bash
python main.py --config config/so3/splitting.yaml
python scripts/nll.py --config config/so3/splitting.yaml --ckpt <ckpt> --N 2048 --eval_refine 5
```

The dataset `dataset/rsgm_SO3_32_batch_100000.npy` (100k rotations from a
32-component mixture in so(3), as in TDM) is included.

### Sec. 4.4 - QM9 conformations on St(n, 4) (Tab. 3, Tab. 4, Tab. 7)

1. Download the processed QM9 splits `qm9_{train,val,test}.pyg` of
   [Stiefel Flow Matching](https://github.com/the-matter-lab/stiefelFM) into
   `dataset/` and build one file per atom count:
   ```bash
   python generate_data/preprocess_qm9_by_size.py --data_dir ./dataset --out_dir ./dataset/qm9_by_size
   ```
2. Train one model per atom count `n` (200k steps each, Tab. 7):
   ```bash
   python main.py --config config/qm9/splitting.yaml --n_atoms 20
   python main.py --config config/qm9/rfm.yaml       --n_atoms 20
   ```
3. Evaluate: NLL, samples and the sample-based energy distance.  The exact
   commands, settings and all raw numbers behind Tab. 3 and Tab. 4 are in
   [`docs/qm9_results.md`](docs/qm9_results.md).

### App. B - inference time (Tab. 6)

```bash
python scripts/measure_qm9_inference_time.py --n_atoms 20 --n_samples 2000 \
    --splitting_ckpt <ckpt> --splitting_eval_refine 5 --rfm_ckpt <ckpt> --rfm_eval_refine 100
```

## Configuration

A run is a YAML file that lists `base_configs` (shared pieces in `config/_base/`)
and overrides a few values; the command line overrides the YAML.  The options
that correspond to symbols of the paper:

| option | paper | meaning |
|---|---|---|
| `T` | T = t_N | terminal time |
| `interval` | N | number of splitting intervals; NFE at inference is N * `eval_refine` |
| `time_grid`, `h_min`, `h_min_steps` | {t_n}, h_n (Def. 1) | `uniform`, or `linear`: `h_min_steps` steps of size `h_min` then linearly growing steps (dense near the data) |
| `velocity_sigma` | sigma | std of the velocity prior; default `1/sqrt(dim M)` |
| `friction_gamma` | gamma_n | OU friction (constant over intervals) |
| `train_bs_x`, `train_bs_t` | M_x, M_t | data points per step and time samples per data point |
| `traj_reuse` | N_reuse | epochs between two refreshes of the cached forward chains |
| `delta_min_ratio` | epsilon | the OU displacement delta is drawn in `[ratio * h_n, h_n]` |
| `eval_refine` | R_n | reverse-OU sub-steps per interval (RFM: total number of ODE steps) |
| `infer_ode` | ODE / SDE mode of Algo. 4 | probability-flow ODE (default) or SDE sampler |
| `potential_sigma` | sigma_x of U(x) = d(x,o)^2 / (2 sigma_x^2) | H^2 only |
| `retraction` | MREI (Def. 2) | `exp` (exact) or `cayley` for SO(n) / St(n,p) |

Run `python main.py --help` for the full list (network, optimizer, logging).

## Implementation notes

* **Score convention.**  The network output, the DSM label and the argument of
  the reverse step are all the sigma^2-scaled vertical score
  `sigma^2 grad_v log p`: this is exactly what the label `-noise / sigma_xi` of
  Eq. (13) estimates when `noise ~ N(0, sigma^2 I)`, so the reverse-step
  coefficients carry no explicit sigma^2 (compare Algo. 4).  The stationary
  score in this convention is `-v`, which is why the default
  `score_parametrization: residual_v` lets the network output `f(x, v, t) - v`.
* **Loss weighting.**  `dsm_loss_weight: local` is the loss written in Algo. 3
  (residual weighted by `sigma_xi(delta) = sqrt(1 - e^{-2 gamma delta})`).  All
  reported runs used `none`, the unweighted MSE on the score, which weights small
  displacements by `1 / sigma_xi^2`; `delta_min_ratio` keeps `delta` away from 0.
* **Time input.**  The network is conditioned on the physical time `t_n + delta`
  of the OU leg (the paper's `tau = 2 t_n + delta` is an equivalent relabelling,
  since the score is only used on OU legs).
* **Potential.**  Compact manifolds use U = 0.  On H^2 the confining potential
  is applied as a symmetric half-kick around the OU and geodesic steps; the
  stationary position law is then a wrapped normal, and the NLL is defined up
  to its (dropped) normalising constant.
* **Validation** is off by default (`--validate` enables NLL and sample plots
  every `check_val_every_n_steps`).  The reported numbers were computed from
  snapshot checkpoints with the scripts in `scripts/`.
* **NLL on St(n,4)** is strongly biased by the inference step count (paper
  Tab. 3); use the sample-based energy distance to compare methods there.

## Adding a manifold

Subclass `manifolds.abstract_manifold.Manifold` and implement the flat
vector <-> natural representation maps, `geodesic_integrator` (exponential map
plus parallel transport of the velocity; it must be momentum-reversal
equivariant, Def. 2), `project_to_tangent_space`, `get_noise`,
`sample_prior_vector` and `log_density_multivariate_normal`; `log_map` is only
needed for the RFM baseline, `potential_grad` / `log_prob_stationary_x` only for
non-compact manifolds.  Register it in `problem_factory.py` with `log Vol(M)`
as `nll_offset`.

## Citation

```bibtex
@inproceedings{kong2026splitting,
  title     = {Enabling Denoising Score Matching Type Training for Manifold Diffusion Model via Momentum and Splitting},
  author    = {Kong, Lingkai and Zhu, Yuchen and Tao, Molei},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
