#!/bin/bash
# AM160 preflight checker for Hummingbird pipeline runs.
#
# Usage:
#   bash hpc/preflight_check.sh
#
# Optional env vars:
#   PROJECT_DIR=/scratch/$USER/AM160_Final
#   CHECK_SBATCH_TEST=1   # run sbatch --test-only checks

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
HPC_DIR="$PROJECT_DIR/hpc"
PROCESSED_DIR="$PROJECT_DIR/results/processed"

required_files=(
  "$PROJECT_DIR/scripts/02_vae_train.py"
  "$PROJECT_DIR/scripts/03_vae_eval.py"
  "$PROJECT_DIR/scripts/04_diffusion_train.py"
  "$PROJECT_DIR/scripts/05_diffusion_eval.py"
  "$HPC_DIR/sbatch_vae_train.sh"
  "$HPC_DIR/sbatch_vae_eval.sh"
  "$HPC_DIR/sbatch_diffusion_train.sh"
  "$HPC_DIR/sbatch_diffusion_eval.sh"
  "$HPC_DIR/run_pipeline.sh"
)

required_processed=(
  "$PROCESSED_DIR/train_pairs.npz"
  "$PROCESSED_DIR/test_pairs_1983.npz"
  "$PROCESSED_DIR/test_pairs_1985.npz"
  "$PROCESSED_DIR/normalization_stats.npz"
  "$PROCESSED_DIR/norm_1983.npy"
  "$PROCESSED_DIR/norm_1985.npy"
)

fail() {
  echo "[FAIL] $1" >&2
  exit 1
}

echo "=== AM160 Preflight Check ==="
echo "Project dir : $PROJECT_DIR"
echo "User        : ${USER:-unknown}"
echo ""

command -v sbatch >/dev/null 2>&1 || fail "sbatch not found. Run on Hummingbird login node."
command -v squeue >/dev/null 2>&1 || fail "squeue not found. Run on Hummingbird login node."

echo "[OK] Slurm commands available"

for f in "${required_files[@]}"; do
  [ -f "$f" ] || fail "Missing required file: $f"
done
echo "[OK] Required scripts present"

for f in "${required_processed[@]}"; do
  [ -f "$f" ] || fail "Missing processed data artifact: $f"
done
echo "[OK] Required processed data present"

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/models" "$PROJECT_DIR/results" "$PROJECT_DIR/plots"
echo "[OK] Output directories ensured"

if grep -RE "[A-Za-z0-9._%+-]+@ucsc\.edu" "$HPC_DIR"/sbatch_*.sh >/dev/null 2>&1; then
  fail "A literal @ucsc.edu address is still present in hpc/sbatch_*.sh. Replace before submit."
fi
echo "[OK] No hard-coded addresses in SLURM scripts"

if [ "${CHECK_SBATCH_TEST:-0}" = "1" ]; then
  (cd "$PROJECT_DIR" && sbatch --test-only hpc/sbatch_vae_train.sh >/dev/null)
  (cd "$PROJECT_DIR" && sbatch --test-only --export=SPARSITY=0.3 hpc/sbatch_diffusion_train.sh >/dev/null)
  echo "[OK] sbatch --test-only checks passed"
fi

echo ""
echo "Preflight passed. Safe to run: bash hpc/run_pipeline.sh"
