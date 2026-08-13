#!/bin/bash
#SBATCH -J vae_eval
#SBATCH -p 96x24gpu4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH -o logs/vae_eval_%j.out

module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
CONDA_ENV="${AM160_CONDA_ENV:-am160}"
conda activate "$CONDA_ENV"

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR"
echo "Working dir: $(pwd)"
echo "Conda env: $CONDA_ENV"
srun python scripts/03_vae_eval.py \
	--data-dir . \
	--out-root . \
	--rollout-steps 40 \
	--rollout-starts 20
