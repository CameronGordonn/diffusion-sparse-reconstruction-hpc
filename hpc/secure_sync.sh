#!/bin/bash
# Secure sync helper for AM160 project assets between local machine and Hummingbird.
#
# Run from LOCAL machine (WSL/macOS/Linux), not on the cluster.
#
# Usage:
#   bash hpc/secure_sync.sh push cruzid
#   bash hpc/secure_sync.sh pull cruzid
#
# Optional env vars:
#   REMOTE_HOST=hb.ucsc.edu
#   REMOTE_BASE=/path/to/AM160_Final   # optional override
#   METHOD=rsync   # rsync (default) or sftp

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: bash hpc/secure_sync.sh <push|pull> cruzid"
  echo "Example: bash hpc/secure_sync.sh push your_cruzid"
  echo "Do not include angle brackets and do not use @ucsc.edu here."
  exit 1
fi

MODE="$1"
CRUZID="$2"
REMOTE_HOST="${REMOTE_HOST:-hb.ucsc.edu}"
REMOTE_BASE="${REMOTE_BASE:-}"
METHOD="${METHOD:-rsync}"

LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: required command not found: $1"
    exit 1
  }
}

check_mode() {
  if [ "$MODE" != "push" ] && [ "$MODE" != "pull" ]; then
    echo "ERROR: mode must be 'push' or 'pull'"
    exit 1
  fi
}

ensure_remote_dirs() {
  ssh "${CRUZID}@${REMOTE_HOST}" "mkdir -p ${REMOTE_BASE}/{results/processed,models,plots,logs,hpc,scripts,data/raw}"
}

resolve_remote_base() {
  if [ -n "$REMOTE_BASE" ]; then
    return
  fi

  # Determine a writable base path on remote: prefer SCRATCH, then common scratch layouts,
  # and finally HOME as a safe fallback.
  REMOTE_BASE="$(ssh "${CRUZID}@${REMOTE_HOST}" '
if [ -n "${SCRATCH:-}" ] && [ -d "${SCRATCH}" ] && [ -w "${SCRATCH}" ]; then
  echo "${SCRATCH}/AM160_Final"
elif [ -d "/scratch/${USER}" ] && [ -w "/scratch/${USER}" ]; then
  echo "/scratch/${USER}/AM160_Final"
elif [ -d "/scratch/users/${USER}" ] && [ -w "/scratch/users/${USER}" ]; then
  echo "/scratch/users/${USER}/AM160_Final"
else
  echo "${HOME}/AM160_Final"
fi
')"

  echo "Resolved remote base: $REMOTE_BASE"
}

push_rsync() {
  ensure_remote_dirs

  rsync -av --progress \
    "$LOCAL_ROOT/results/processed/" \
    "${CRUZID}@${REMOTE_HOST}:${REMOTE_BASE}/results/processed/"

  rsync -av --progress \
    --include='*/' \
    --include='*.py' \
    --include='*.md' \
    --exclude='*' \
    "$LOCAL_ROOT/" \
    "${CRUZID}@${REMOTE_HOST}:${REMOTE_BASE}/"

  rsync -av --progress \
    "$LOCAL_ROOT/hpc/" \
    "${CRUZID}@${REMOTE_HOST}:${REMOTE_BASE}/hpc/"
}

pull_rsync() {
  mkdir -p "$LOCAL_ROOT/models" "$LOCAL_ROOT/plots" "$LOCAL_ROOT/results"

  rsync -av --progress \
    "${CRUZID}@${REMOTE_HOST}:${REMOTE_BASE}/models/" \
    "$LOCAL_ROOT/models/"

  rsync -av --progress \
    "${CRUZID}@${REMOTE_HOST}:${REMOTE_BASE}/plots/" \
    "$LOCAL_ROOT/plots/"

  rsync -av --progress \
    "${CRUZID}@${REMOTE_HOST}:${REMOTE_BASE}/results/" \
    "$LOCAL_ROOT/results/"
}

push_sftp() {
  ensure_remote_dirs

  tmpfile="$(mktemp)"
  cat > "$tmpfile" <<EOF
put $LOCAL_ROOT/results/processed/train_pairs.npz $REMOTE_BASE/results/processed/
put $LOCAL_ROOT/results/processed/test_pairs_1983.npz $REMOTE_BASE/results/processed/
put $LOCAL_ROOT/results/processed/test_pairs_1985.npz $REMOTE_BASE/results/processed/
put $LOCAL_ROOT/results/processed/normalization_stats.npz $REMOTE_BASE/results/processed/
put $LOCAL_ROOT/results/processed/norm_1979.npy $REMOTE_BASE/results/processed/
put $LOCAL_ROOT/results/processed/norm_1980.npy $REMOTE_BASE/results/processed/
put $LOCAL_ROOT/results/processed/norm_1981.npy $REMOTE_BASE/results/processed/
put $LOCAL_ROOT/results/processed/norm_1983.npy $REMOTE_BASE/results/processed/
put $LOCAL_ROOT/results/processed/norm_1985.npy $REMOTE_BASE/results/processed/
put $LOCAL_ROOT/scripts/01_data_loading.py $REMOTE_BASE/scripts/
put $LOCAL_ROOT/scripts/02_vae_model.py $REMOTE_BASE/scripts/
put $LOCAL_ROOT/scripts/02_vae_train.py $REMOTE_BASE/scripts/
put $LOCAL_ROOT/scripts/03_vae_eval.py $REMOTE_BASE/scripts/
put $LOCAL_ROOT/scripts/04_diffusion_model_minimal.py $REMOTE_BASE/scripts/
put $LOCAL_ROOT/scripts/04_diffusion_model_v2.py $REMOTE_BASE/scripts/
put $LOCAL_ROOT/scripts/04_diffusion_train.py $REMOTE_BASE/scripts/
put $LOCAL_ROOT/scripts/05_diffusion_eval.py $REMOTE_BASE/scripts/
put $LOCAL_ROOT/hpc/sbatch_vae_train.sh $REMOTE_BASE/hpc/
put $LOCAL_ROOT/hpc/sbatch_diffusion_train.sh $REMOTE_BASE/hpc/
put $LOCAL_ROOT/hpc/sbatch_vae_eval.sh $REMOTE_BASE/hpc/
put $LOCAL_ROOT/hpc/sbatch_diffusion_eval.sh $REMOTE_BASE/hpc/
put $LOCAL_ROOT/hpc/run_pipeline.sh $REMOTE_BASE/hpc/
put $LOCAL_ROOT/hpc/preflight_check.sh $REMOTE_BASE/hpc/
EOF

  sftp -b "$tmpfile" "${CRUZID}@${REMOTE_HOST}"
  rm -f "$tmpfile"
}

pull_sftp() {
  mkdir -p "$LOCAL_ROOT/models" "$LOCAL_ROOT/plots" "$LOCAL_ROOT/results"

  tmpfile="$(mktemp)"
  cat > "$tmpfile" <<EOF
get -r $REMOTE_BASE/models/* $LOCAL_ROOT/models/
get -r $REMOTE_BASE/plots/* $LOCAL_ROOT/plots/
get -r $REMOTE_BASE/results/* $LOCAL_ROOT/results/
EOF

  sftp -b "$tmpfile" "${CRUZID}@${REMOTE_HOST}"
  rm -f "$tmpfile"
}

main() {
  check_mode

  require_cmd ssh
  resolve_remote_base

  if [ "$METHOD" = "rsync" ]; then
    require_cmd rsync
    if [ "$MODE" = "push" ]; then
      push_rsync
    else
      pull_rsync
    fi
  elif [ "$METHOD" = "sftp" ]; then
    require_cmd sftp
    if [ "$MODE" = "push" ]; then
      push_sftp
    else
      pull_sftp
    fi
  else
    echo "ERROR: METHOD must be 'rsync' or 'sftp'"
    exit 1
  fi

  echo "Secure sync completed: mode=$MODE method=$METHOD"
}

main
