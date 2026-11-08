#!/usr/bin/env bash
# Train one model on one GPU.
#
#   export MLFLOW_TRACKING_URI=file:///<shared>/mlruns
#   sbatch slurm/train.sbatch configs/<run>.json
#
# Adapt partition, memory and time to your cluster and dataset.
#SBATCH --job-name=sbt-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
CONFIG=${1:?usage: sbatch slurm/train.sbatch CONFIG}
: "${MLFLOW_TRACKING_URI:?export MLFLOW_TRACKING_URI=file:///<shared>/mlruns}"
: "${SBT_CONDA_ENV:=seisbench-trainer}"

# Shared folder store (decision D-1); MLflow refuses it without this opt-out.
export MLFLOW_ALLOW_FILE_STORE=true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SBT_CONDA_ENV"

srun seisbench-trainer train "$CONFIG"
