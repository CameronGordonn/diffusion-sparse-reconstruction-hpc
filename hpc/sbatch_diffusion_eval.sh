#!/bin/bash
#SBATCH -J diff_eval
#SBATCH -p 96x24gpu4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH -o logs/diff_eval_%j.out

module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
CONDA_ENV="${AM160_CONDA_ENV:-am160}"
conda activate "$CONDA_ENV"

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR"
echo "Working dir: $(pwd)"
echo "Conda env: $CONDA_ENV"
MODEL_TYPE="${MODEL_TYPE:-minimal}"
echo "MODEL_TYPE: $MODEL_TYPE"
# --num-samples 20: more stable ensemble mean/std for uncertainty reporting
# --num-cases 20: evaluates more test snapshots now that we have GPU
srun python scripts/05_diffusion_eval.py --data-dir . --out-root . --model-type "$MODEL_TYPE" --num-samples 20 --num-cases 20
