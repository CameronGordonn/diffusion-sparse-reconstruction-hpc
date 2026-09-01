# Generative Forecasting and Sparse Reconstruction of Geopotential Height Fields

Deep generative models for geopotential height fields (z500, z50), trained on GPU nodes of
the **UCSC Hummingbird** cluster via SLURM. AM160 final project, Winter 2026.

**Author:** Cameron Gordon

Two problems:

- **Phase 1 — Conditional VAE** for 6-hour-ahead forecasting, evaluated one-step and
  autoregressively out to 72 hours.
- **Phase 2 — Conditional diffusion** for reconstructing a full field from sparse
  observations, at 30%, 60%, and 90% masking.

Data is ERA5 geopotential, one NetCDF file per year (1979–1985), shape `(T, 2, H, W)`
where the size-2 axis is the two pressure levels.

## Repository layout

```
scripts/     data loading, VAE + diffusion models, training and evaluation
hpc/         SLURM job scripts and the pipeline orchestrator
tools/       environment checks and model-initialization smoke tests
docs/        report PDF, planning guides, auto-generated results tables
plots/       figures for each phase
results/     evaluation metrics (JSON)
models/      training histories (checkpoints ship as a GitHub Release, not in git)
```

## Running on Hummingbird

[`HUMMINGBIRD.md`](HUMMINGBIRD.md) has the full walkthrough. The short version, from a
login node:

```bash
bash hpc/preflight_check.sh     # verify scripts, data, and SLURM availability
bash hpc/run_pipeline.sh        # submit everything with dependency chaining
squeue -u $USER                 # monitor
```

`run_pipeline.sh` submits the VAE training job and three diffusion training jobs (one per
sparsity level) in parallel, then gates the two evaluation jobs behind them with
`--dependency=afterok`. Jobs request one GPU on the `96x24gpu4` partition.

Individual jobs can be submitted directly; the diffusion script requires a sparsity level:

```bash
sbatch --export=SPARSITY=0.6,MODEL_TYPE=v2 hpc/sbatch_diffusion_train.sh
```

## Results

Full tables in [`docs/guides/FINAL_NUMBERS.md`](docs/guides/FINAL_NUMBERS.md); the write-up
is [`docs/AM160_Final_Report.pdf`](docs/AM160_Final_Report.pdf).

### Phase 1 — Forecasting: the VAE has no skill

Stated plainly, because the original write-up did not: **the forecaster loses to every
trivial baseline at every lead time.** Normalized RMSE, mean of the two test years:

| Lead | Persistence | Per-gridpoint climatology | Global training mean | **VAE** |
|---|---:|---:|---:|---:|
| 6 h | **0.057** | 0.59 | ~1.0 | **1.09** |
| 24 h | 0.154 | 0.59 | ~1.0 | 0.93 |
| 72 h | 0.266 | 0.59 | ~1.0 | 0.94 |
| 240 h | 0.345 | 0.59 | ~1.0 | 0.93 |

At 6 hours the VAE is ~19× worse than copying the input forward, and it never beats a
single global constant at any lead. Persistence at 10 days is still three times better
than the VAE at 6 hours.

The flat rollout curve is the diagnosis, not a feature. Normalization is per-channel and
global, so predicting zero *is* predicting the training mean and scores ~1.0 by
construction. A curve that starts at 1.09 and settles at ~0.93 is a model that collapses
to near-climatology immediately and stays there — error saturates on the first step and
has nowhere left to grow. A forecaster with skill instead starts low and rises
monotonically, crossing climatology at some finite lead, which is exactly what the
persistence curve does.

This is unfixed. The evaluation now reports and plots the baselines so the gap is legible;
diagnosing the mean collapse itself (likely posterior collapse in the latent, given
`beta` and the reconstruction/KL balance) is future work.

**Diffusion reconstruction RMSE (normalized), cross-year means:**

| Sparsity | Unobserved | Observed |
|---|---:|---:|
| 30% | 152.07 | 1.12e-07 |
| 60% | 214.17 | 1.12e-07 |
| 90% | 257.80 | 1.12e-07 |

Observed-pixel error sits at floating-point noise, confirming the conditioning mechanism
holds observed values exactly. Unobserved-region error is large in normalized units — see
the report for discussion.

## Reproducing

Data and processed arrays are not tracked (~3.2 GB combined). To regenerate:

1. Place the yearly ERA5 NetCDF files in `data/raw/` as `z1979.nc` … `z1985.nc`.
2. Run `python scripts/01_data_loading.py` to write normalized arrays and train/test pairs
   into `results/processed/`.
3. Run `bash hpc/preflight_check.sh` to confirm every required artifact is in place.

## Checkpoints

`models/` holds both diffusion architectures at all three sparsity levels:

| File | Architecture | Notes |
|---|---|---|
| `diffusion_{30,60,90}pct_v2_best.pt` | `v2` | **Produced the reported metrics** |
| `diffusion_{30,60,90}pct_best.pt` | `minimal` | Earlier baseline run |

Evaluate against either by passing the matching `--model-type`; the loader also reads the
architecture recorded inside the checkpoint itself:

```bash
python scripts/05_diffusion_eval.py --data-dir . --out-root . --model-type v2
```

The **VAE checkpoint is not included** — at 176 MB it exceeds GitHub's 100 MB per-file
limit. Retrain it with `hpc/sbatch_vae_train.sh`.
