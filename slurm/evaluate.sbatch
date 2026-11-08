#!/usr/bin/env bash
# Evaluate a model on a dataset split on one GPU; arguments go to
# `seisbench-trainer evaluate`.
#
#   export MLFLOW_TRACKING_URI=file:///<shared>/mlruns
#   sbatch slurm/evaluate.sbatch --model <run_id or models:/...> [--dataset NAME] [--split test]
#
# Only the dev split (for thresholds) and the evaluated split are preloaded.
#SBATCH --job-name=sbt-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
: "${MLFLOW_TRACKING_URI:?export MLFLOW_TRACKING_URI=file:///<shared>/mlruns}"
: "${SBT_CONDA_ENV:=seisbench-trainer}"
export MLFLOW_ALLOW_FILE_STORE=true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SBT_CONDA_ENV"

srun seisbench-trainer evaluate "$@"
