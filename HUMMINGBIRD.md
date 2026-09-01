# Running on UCSC Hummingbird

## Quick Reference

| Step | Where | Command |
|---|---|---|
| Login | local terminal | `ssh <cruzid>@hb.ucsc.edu` |
| Create workspace | cluster login node | `mkdir -p /scratch/$USER/AM160_Final/{results/processed,models,plots/phase2_diffusion,logs}` |
| Transfer data (secure) | local terminal | `bash hpc/secure_sync.sh push <cruzid>` |
| Build conda env | cluster login node | see §5 below |
| Submit all jobs | cluster login node | `bash hpc/run_pipeline.sh` |
| Monitor | cluster login node | `squeue -u $USER` |

---

## 1. Prerequisites

- UCSC CruzID and Gold password
- Off-campus: connect to [UCSC VPN](https://its.ucsc.edu/vpn/) **before** SSH

---

## 2. SSH Login

```bash
ssh <your_cruzid>@hb.ucsc.edu
```

---

## 3. Create Scratch Workspace (on the cluster)

```bash
mkdir -p /scratch/$USER/AM160_Final/{results/processed,models,plots/phase2_diffusion,logs}
```

## 4. Transfer Files (from your local machine)

Recommended secure transfer (WSL/macOS/Linux):

```bash
cd /path/to/your/local/checkout

# Preferred: rsync over SSH (incremental and resumable)
bash hpc/secure_sync.sh push your_cruzid

# Alternative: force SFTP batch mode if rsync is unavailable
METHOD=sftp bash hpc/secure_sync.sh push your_cruzid
```

This helper script only uses encrypted SSH transport (`rsync` over SSH or `sftp`).
Pass only your CruzID token (for example `your_cruzid`), not `@ucsc.edu` and not angle brackets.

### Option A — sftp command line (Mac/Linux/WSL)

```bash
# Replace <cruzid> with your actual CruzID everywhere below

# 1. Transfer pre-processed data arrays (skips re-running Phase 0 on the cluster)
sftp <cruzid>@hb.ucsc.edu << 'EOF'
put results/processed/train_pairs.npz /scratch/<cruzid>/AM160_Final/results/processed/
put results/processed/test_pairs_1983.npz /scratch/<cruzid>/AM160_Final/results/processed/
put results/processed/test_pairs_1985.npz /scratch/<cruzid>/AM160_Final/results/processed/
put results/processed/normalization_stats.npz /scratch/<cruzid>/AM160_Final/results/processed/
put results/processed/norm_1979.npy /scratch/<cruzid>/AM160_Final/results/processed/
put results/processed/norm_1980.npy /scratch/<cruzid>/AM160_Final/results/processed/
put results/processed/norm_1981.npy /scratch/<cruzid>/AM160_Final/results/processed/
put results/processed/norm_1983.npy /scratch/<cruzid>/AM160_Final/results/processed/
put results/processed/norm_1985.npy /scratch/<cruzid>/AM160_Final/results/processed/
EOF

# 2. Transfer Python scripts
sftp <cruzid>@hb.ucsc.edu << 'EOF'
mkdir /scratch/<cruzid>/AM160_Final/scripts
put scripts/01_data_loading.py /scratch/<cruzid>/AM160_Final/scripts/
put scripts/02_vae_model.py /scratch/<cruzid>/AM160_Final/scripts/
put scripts/02_vae_train.py /scratch/<cruzid>/AM160_Final/scripts/
put scripts/03_vae_eval.py /scratch/<cruzid>/AM160_Final/scripts/
put scripts/04_diffusion_model_minimal.py /scratch/<cruzid>/AM160_Final/scripts/
put scripts/04_diffusion_model_v2.py /scratch/<cruzid>/AM160_Final/scripts/
put scripts/04_diffusion_train.py /scratch/<cruzid>/AM160_Final/scripts/
put scripts/05_diffusion_eval.py /scratch/<cruzid>/AM160_Final/scripts/
EOF

# 3. Transfer HPC scripts
sftp <cruzid>@hb.ucsc.edu << 'EOF'
mkdir /scratch/<cruzid>/AM160_Final/hpc
put hpc/sbatch_vae_train.sh /scratch/<cruzid>/AM160_Final/hpc/
put hpc/sbatch_diffusion_train.sh /scratch/<cruzid>/AM160_Final/hpc/
put hpc/sbatch_vae_eval.sh /scratch/<cruzid>/AM160_Final/hpc/
put hpc/sbatch_diffusion_eval.sh /scratch/<cruzid>/AM160_Final/hpc/
put hpc/run_pipeline.sh /scratch/<cruzid>/AM160_Final/hpc/
EOF

---

## 5. Build Conda Environment (on the cluster, once only)

```bash
module load miniconda3
conda create -n am160 python=3.11 -y
conda activate base
conda install pytorch torchvision -c pytorch -y
pip install numpy scipy matplotlib xarray netCDF4
```

Verify GPU is accessible:
```bash
salloc --partition=96x24gpu4 --gres=gpu:1 --mem=4G --time=0:05:00
# once allocated:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
exit
```

## 6. Set Your Email in SLURM Scripts

Replace the placeholder email so Slurm can notify you on job completion or failure:

```bash
cd /scratch/$USER/AM160_Final
sed -i 's/REPLACE_WITH_YOUR_CRUZID/<your_cruzid>/g' hpc/*.sh
```

---

## 7. Make Scripts Executable

```bash
chmod +x /scratch/$USER/AM160_Final/hpc/*.sh
```

---

## 8. Submit the Full Pipeline

```bash
cd /scratch/$USER/AM160_Final
bash hpc/run_pipeline.sh
```

Output will list all 6 job IDs:

```
=== AM160 Pipeline Submission ===
VAE train submitted        : job 12001
Diffusion 30% submitted    : job 12002
Diffusion 60% submitted    : job 12003
Diffusion 90% submitted    : job 12004
VAE eval submitted         : job 12005  (waits for 12001)
Diffusion eval submitted   : job 12006  (waits for 12002, 12003, 12004)
```

The 3 diffusion training jobs run **in parallel** with each other and the VAE train. Eval jobs start automatically once training succeeds.

---

## 9. Monitor Jobs

```bash
squeue -u $USER                            # Live queue status for all your jobs
scontrol show job <JID>                    # Detailed info for one job
tail -f logs/vae_train_<JID>.out          # Stream training output live
sacct -o reqmem,maxrss,elapsed -j <JID>   # Post-run memory/time usage
```

Job states: `PD` = pending (waiting for resources), `R` = running, `CG` = completing, `CD` = completed

---

## 10. Retrieve Results (from your local machine)

Recommended secure pull:

```bash
cd /path/to/your/local/checkout

# Preferred pull path
bash hpc/secure_sync.sh pull your_cruzid

# SFTP fallback
METHOD=sftp bash hpc/secure_sync.sh pull your_cruzid
```

Manual SFTP pull (equivalent):

```bash
sftp <cruzid>@hb.ucsc.edu << 'EOF'
get -r /scratch/<cruzid>/AM160_Final/models ./models_hb
get -r /scratch/<cruzid>/AM160_Final/plots ./plots_hb
get -r /scratch/<cruzid>/AM160_Final/results ./results_hb
EOF
```
---

## 13. GPU Partition Reference

| Partition | Nodes | GPUs | RAM | Cores/Node |
|---|---|---|---|---|
| `96x24gpu4` | 1 | 4× Tesla P100 | 96 GB | 24 |
| `128x24` | ~19 | None | 128 GB | 24 |
| `256x44` | 1 | None | 256 GB | 44 |

All jobs in this project use `96x24gpu4`.

---
