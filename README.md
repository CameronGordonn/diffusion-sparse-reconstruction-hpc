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
models/      diffusion checkpoints and training histories
logs/        SLURM job output from the cluster runs
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

**VAE one-step RMSE (normalized):** 1.103 (1983), 1.071 (1985). Autoregressive rollout
stays near 0.91–0.97 through 72 hours.

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
