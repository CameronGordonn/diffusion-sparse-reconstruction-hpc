#!/bin/bash
# AM160 Final — Hummingbird Pipeline Orchestrator
#
# Submits all training and evaluation jobs to Slurm with dependency chaining.
# VAE train + 3 diffusion train jobs are submitted immediately (in parallel).
# Eval jobs are gated behind their respective training jobs (afterok).
#
# Usage: run from the LOGIN NODE inside your scratch project directory
#   cd /scratch/$USER/AM160_Final
#   bash hpc/run_pipeline.sh
#
# Requires:
#   - hpc/sbatch_*.sh scripts present and executable
#   - logs/ directory exists (created automatically below)
#   - conda env 'am160' already built (see HUMMINGBIRD.md)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

require_file() {
    local f="$1"
    if [ ! -f "$f" ]; then
        echo "ERROR: Required file missing: $f" >&2
        exit 1
    fi
}

# Ensure logs directory exists
mkdir -p "$PROJECT_DIR/logs"

if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch not found. Run this on a Hummingbird login node." >&2
    exit 1
fi

require_file "$SCRIPT_DIR/sbatch_vae_train.sh"
require_file "$SCRIPT_DIR/sbatch_vae_eval.sh"
require_file "$SCRIPT_DIR/sbatch_diffusion_train.sh"
require_file "$SCRIPT_DIR/sbatch_diffusion_eval.sh"

if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
    require_file "$SCRIPT_DIR/preflight_check.sh"
    echo "Running preflight checks..."
    if ! PROJECT_DIR="$PROJECT_DIR" bash "$SCRIPT_DIR/preflight_check.sh"; then
        echo "ERROR: Preflight failed. Fix issues above and rerun." >&2
        echo "Hint: set SKIP_PREFLIGHT=1 only if you have already validated manually." >&2
        exit 1
    fi
    echo "Preflight passed. Proceeding to submission."
    echo ""
fi

echo "=== AM160 Pipeline Submission ==="
echo "Project dir : $PROJECT_DIR"
echo "SLURM user  : $USER"
echo ""

# cd into PROJECT_DIR so that SLURM_SUBMIT_DIR is set to PROJECT_DIR for all jobs
cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# Training jobs — all start immediately
# ---------------------------------------------------------------------------

VAEJID=$(sbatch --parsable --export=ALL "$SCRIPT_DIR/sbatch_vae_train.sh")
echo "VAE train submitted        : job $VAEJID"

D30JID=$(sbatch --parsable --export=ALL,SPARSITY=0.3,MODEL_TYPE=${MODEL_TYPE:-minimal},BASE_CHANNELS=${BASE_CHANNELS:-64},USE_AMP=${USE_AMP:-1} "$SCRIPT_DIR/sbatch_diffusion_train.sh")
echo "Diffusion 30% submitted    : job $D30JID"

D60JID=$(sbatch --parsable --export=ALL,SPARSITY=0.6,MODEL_TYPE=${MODEL_TYPE:-minimal},BASE_CHANNELS=${BASE_CHANNELS:-64},USE_AMP=${USE_AMP:-1} "$SCRIPT_DIR/sbatch_diffusion_train.sh")
echo "Diffusion 60% submitted    : job $D60JID"

D90JID=$(sbatch --parsable --export=ALL,SPARSITY=0.9,MODEL_TYPE=${MODEL_TYPE:-minimal},BASE_CHANNELS=${BASE_CHANNELS:-64},USE_AMP=${USE_AMP:-1} "$SCRIPT_DIR/sbatch_diffusion_train.sh")
echo "Diffusion 90% submitted    : job $D90JID"

# ---------------------------------------------------------------------------
# Evaluation jobs — gated on training success
# ---------------------------------------------------------------------------

VAEEVALID=$(sbatch --parsable \
    --export=ALL \
    --dependency=afterok:${VAEJID} \
    "$SCRIPT_DIR/sbatch_vae_eval.sh")
echo "VAE eval submitted         : job $VAEEVALID  (waits for $VAEJID)"

DIFFEVALID=$(sbatch --parsable \
    --export=ALL \
    --dependency=afterok:${D30JID}:${D60JID}:${D90JID} \
    "$SCRIPT_DIR/sbatch_diffusion_eval.sh")
echo "Diffusion eval submitted   : job $DIFFEVALID  (waits for $D30JID, $D60JID, $D90JID)"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== All jobs submitted ==="
echo ""
echo "Monitor        : squeue -u \$USER"
echo "Stream VAE log : tail -f $PROJECT_DIR/logs/vae_train_${VAEJID}.out"
echo "Cancel all     : scancel $VAEJID $D30JID $D60JID $D90JID $VAEEVALID $DIFFEVALID"
echo ""
echo "After completion, retrieve results with:"
echo "  sftp $USER@hb.ucsc.edu"
echo "  get -r $PROJECT_DIR/models"
echo "  get -r $PROJECT_DIR/plots"
echo "  get -r $PROJECT_DIR/results"
