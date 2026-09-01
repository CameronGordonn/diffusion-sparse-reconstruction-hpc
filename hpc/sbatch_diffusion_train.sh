#!/bin/bash
#SBATCH -J diff_train
#SBATCH -p 96x24gpu4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH -o logs/diff_train_%j.out

set -euo pipefail

# SPARSITY is passed via --export=SPARSITY=0.3|0.6|0.9 at sbatch submission time.
# Example: sbatch --export=SPARSITY=0.3 hpc/sbatch_diffusion_train.sh
if [ -z "${SPARSITY:-}" ]; then
    echo "ERROR: SPARSITY env var not set."
    echo "Usage: sbatch --export=SPARSITY=0.3 hpc/sbatch_diffusion_train.sh"
    exit 1
fi

module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
CONDA_ENV="${AM160_CONDA_ENV:-am160}"
conda activate "$CONDA_ENV"

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR"
echo "Working dir: $(pwd)"
echo "Conda env: $CONDA_ENV"
echo "Python path: $(which python)"
echo "SPARSITY: $SPARSITY"
MODEL_TYPE="${MODEL_TYPE:-minimal}"
MASK_TYPE="${MASK_TYPE:-iid}"
MASK_BLOCK="${MASK_BLOCK:-8}"
BASE_CHANNELS="${BASE_CHANNELS:-64}"
USE_AMP="${USE_AMP:-1}"
echo "MODEL_TYPE: $MODEL_TYPE"
echo "MASK_TYPE: $MASK_TYPE (block=$MASK_BLOCK)"
echo "BASE_CHANNELS: $BASE_CHANNELS"
echo "USE_AMP: $USE_AMP"
if [ ! -f "$PROJECT_DIR/scripts/04_diffusion_train.py" ]; then
    echo "ERROR: Missing training script at $PROJECT_DIR/scripts/04_diffusion_train.py"
    exit 2
fi

AMP_FLAG="--use-amp"
if [ "$USE_AMP" = "0" ]; then
    AMP_FLAG="--no-use-amp"
fi

srun python "$PROJECT_DIR/scripts/04_diffusion_train.py" \
    --data-dir "$PROJECT_DIR" \
    --out-root "$PROJECT_DIR" \
    --model-type "$MODEL_TYPE" \
    --base-channels "$BASE_CHANNELS" \
    --sparsity "$SPARSITY" \
    --mask-type "$MASK_TYPE" \
    --mask-block "$MASK_BLOCK" \
    --epochs 100 \
    --patience 25 \
    --lr 5e-5 \
    $AMP_FLAG
