#!/usr/bin/env bash
# HPO worker array: every task joins the same Optuna study and trains trials
# on one GPU until the study has hpo.n_trials finished trials.
#
#   export MLFLOW_TRACKING_URI=file:///<shared>/mlruns
#   sbatch --array=0-3 slurm/hpo_worker.sbatch configs/<hpo>.json
#
# hpo.storage (the Optuna journal) must be on the shared filesystem.
#SBATCH --job-name=sbt-hpo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail
CONFIG=${1:?usage: sbatch --array=0-N slurm/hpo_worker.sbatch CONFIG}
: "${MLFLOW_TRACKING_URI:?export MLFLOW_TRACKING_URI=file:///<shared>/mlruns}"
: "${SBT_CONDA_ENV:=seisbench-trainer}"
export MLFLOW_ALLOW_FILE_STORE=true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SBT_CONDA_ENV"

srun seisbench-trainer hpo "$CONFIG"
