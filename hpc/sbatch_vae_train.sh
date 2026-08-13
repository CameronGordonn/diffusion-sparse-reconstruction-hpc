#!/bin/bash
#SBATCH -J vae_train
#SBATCH -p 96x24gpu4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH -o logs/vae_train_%j.out

module load miniconda3
# Source conda shell hooks — required for 'conda activate' in non-interactive SLURM scripts
source "$(conda info --base)/etc/profile.d/conda.sh"
CONDA_ENV="${AM160_CONDA_ENV:-am160}"
conda activate "$CONDA_ENV"

# SLURM_SUBMIT_DIR = the directory from which sbatch was invoked (set automatically by Slurm)
# We fall back to it so the job always runs from the project root regardless of how the
# script was copied into the Slurm spool directory.
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR"
echo "Working dir: $(pwd)"
echo "Conda env: $CONDA_ENV"
srun python scripts/02_vae_train.py --data-dir . --out-root .
