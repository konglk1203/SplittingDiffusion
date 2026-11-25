# QM9 on the Stiefel manifold: evaluation protocol and raw numbers

Source material for Sec. 4.4 (Tab. 3, Tab. 4) and App. B (Tab. 6) of the paper.
One model per atom count n (molecules as points of St(n, 4), see
`generate_data/preprocess_qm9_by_size.py`), trained for 200,000 steps with
`config/qm9/splitting.yaml` and `config/qm9/rfm.yaml` (hyperparameters: paper Tab. 7).

## Evaluation protocol

| quantity | tool | setting |
|---|---|---|
| NLL (nats) | `scripts/nll.py` | 2048 real molecules; splitting `--eval_refine R` (NFE = 20 R), RFM `--eval_refine NFE` |
| samples | `scripts/sample.py` | 5000 molecules, ODE sampler for both methods; splitting `--eval_refine 10` (NFE 200), RFM `--eval_refine 200` |
| energy distance | `scripts/qm9_sample_metric.py` | 2000 generated vs 2000 real molecules, floor from two disjoint halves of the real data (`--n 2000`) |

```bash
# n = 20; replace the checkpoint paths by your runs
python scripts/nll.py    --config config/qm9/splitting.yaml --n_atoms 20 --eval_refine 10  --ckpt SP.ckpt  --N 2048
python scripts/nll.py    --config config/qm9/rfm.yaml       --n_atoms 20 --eval_refine 200 --ckpt RFM.ckpt --N 2048
python scripts/sample.py --config config/qm9/splitting.yaml --n_atoms 20 --eval_refine 10  --ckpt SP.ckpt  --out samples/sp20.npy  --N 5000
python scripts/sample.py --config config/qm9/rfm.yaml       --n_atoms 20 --eval_refine 200 --ckpt RFM.ckpt --out samples/rfm20.npy --N 5000
python scripts/qm9_sample_metric.py --data dataset/qm9_by_size/qm9_N020.npy --N 20 --n 2000 \
    --gen splitting=samples/sp20.npy --gen rfm=samples/rfm20.npy
```

The real molecules of the reported numbers were drawn from the whole per-n file
(`--split all`, the default of the two scripts).  Pass `--split heldout` to
restrict both tools to the 10% validation split of `main.py`.

## NLL is not a usable ranking metric here (paper Tab. 3)

n = 20, final checkpoints, 2048 real molecules.  NLL moves by hundreds of nats
with the inference step count and is still moving at NFE = 400, so it does not
have a stable value; it also changes the ranking of the two methods.

| NFE | splitting (R = NFE/20) | RFM |
|---:|---:|---:|
| 20  | -1018.13 | -951.29 |
| 100 | -445.99  | -481.24 |
| 200 | -389.60  | -434.74 |
| 400 | -368.38  | -410.85 |

NLL at NFE 200 for every atom count (same caveat):

| n | splitting | RFM |
|---:|---:|---:|
| 15 | -274.07 | -294.44 |
| 16 | -280.29 | -315.47 |
| 17 | -295.06 | -343.55 |
| 18 | -319.93 | -367.47 |
| 19 | -344.18 | -405.02 |
| 20 | -389.79 | -432.61 |
| 21 | -404.44 | -470.25 |
| 22 | -440.89 | -499.77 |

## Sample-based energy distance (paper Tab. 4)

Lower is better; TOTAL is the sum of the four descriptors.  Both methods at
NFE = 200 (splitting 20 x 10, RFM 200), 2000 vs 2000 molecules.

| n | set | pdist | mass | massw_dist | shape | TOTAL |
|---:|---|---:|---:|---:|---:|---:|
| 15 | floor | 0.0135 | 0.0077 | 0.0129 | 0.0017 | 0.0359 |
|    | splitting | 0.0221 | 0.2266 | 0.0549 | 0.2185 | **0.5221** |
|    | RFM | 0.0352 | 0.2894 | 0.0507 | 0.2315 | 0.6067 |
| 16 | floor | 0.0117 | 0.0046 | 0.0171 | 0.0004 | 0.0338 |
|    | splitting | 0.0409 | 0.3452 | 0.0713 | 0.2124 | **0.6697** |
|    | RFM | 0.0303 | 0.4254 | 0.0563 | 0.2124 | 0.7245 |
| 17 | floor | 0.0183 | 0.0029 | 0.0176 | 0.0009 | 0.0397 |
|    | splitting | 0.0773 | 0.2697 | 0.0371 | 0.1993 | **0.5834** |
|    | RFM | 0.0352 | 0.4155 | 0.0319 | 0.1926 | 0.6752 |
| 18 | floor | 0.0203 | 0.0093 | 0.0180 | 0.0010 | 0.0485 |
|    | splitting | 0.0724 | 0.2895 | 0.0551 | 0.1786 | **0.5955** |
|    | RFM | 0.0570 | 0.5390 | 0.0466 | 0.1579 | 0.8006 |
| 19 | floor | 0.0172 | 0.0160 | 0.0240 | 0.0013 | 0.0585 |
|    | splitting | 0.0285 | 0.3947 | 0.0537 | 0.1758 | **0.6527** |
|    | RFM | 0.0547 | 0.5608 | 0.0579 | 0.1723 | 0.8456 |
| 20 | floor | 0.0229 | 0.0085 | 0.0275 | 0.0042 | 0.0631 |
|    | splitting | 0.0478 | 0.7293 | 0.0670 | 0.1456 | **0.9896** |
|    | RFM | 0.0953 | 0.9150 | 0.1293 | 0.1534 | 1.2930 |
| 21 | floor | 0.0197 | 0.0015 | 0.0193 | 0.0014 | 0.0418 |
|    | splitting | 0.1021 | 0.5066 | 0.0729 | 0.1397 | **0.8212** |
|    | RFM | 0.0438 | 0.9859 | 0.0822 | 0.1388 | 1.2506 |
| 22 | floor | 0.0391 | 0.0033 | 0.0313 | 0.0028 | 0.0765 |
|    | splitting | 0.0780 | 1.2782 | 0.1100 | 0.1317 | **1.5978** |
|    | RFM | 0.1328 | 1.7215 | 0.2749 | 0.1269 | 2.2561 |

Splitting has the lower TOTAL at every n; most of the margin comes from the
mass descriptor.  The number of real molecules per n (Tab. 4, "# Data") is
8374, 11359, 13761, 14192, 14648, 10060, 10554, 3629 for n = 15, ..., 22.

## Training time (paper App. B)

n = 20, 200,000 steps, one V100, validation disabled (first and last log
timestamps):

| method | wall-clock | per step |
|---|---:|---:|
| splitting (`train_bs_x` 64, `train_bs_t` 16) | 14 h 54 min | 0.27 s |
| RFM (`train_bs_x` 128) | 40 h 54 min | 0.74 s |

RFM's step is dominated by the Stiefel logarithm map (one iterative solve per
molecule); splitting only evaluates the exponential map when the forward chains
are refreshed.  Inference time per sample is measured with
`scripts/measure_qm9_inference_time.py`.
