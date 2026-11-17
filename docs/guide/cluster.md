# Slurm cluster

The scripts in `slurm/` run one job per GPU with a conda environment and an
MLflow folder store on the shared filesystem; no MLflow server is needed.

```bash
export MLFLOW_TRACKING_URI=file:///shared/mlruns
sbatch slurm/train.sbatch configs/run.json
sbatch slurm/evaluate.sbatch --model <run_id>
sbatch --array=0-3 slurm/hpo_worker.sbatch configs/hpo.json
```

The scripts set `MLFLOW_ALLOW_FILE_STORE=true`, which MLflow 3.18 requires
for a folder store, and activate the conda env `$SBT_CONDA_ENV` (default
`seisbench-trainer`). Adapt partition, memory and time to your cluster: a run
preloads all splits into memory.

To browse the results, start the UI on a login node and open it through an
SSH tunnel:

```bash
mlflow ui --backend-store-uri file:///shared/mlruns --port 5000
```

## Notes from running it

- **Install editable and reinstall after every commit**
  (`pip install -e . --no-deps`): the version, which names the commit, is
  fixed at install time.
- **Jobs run the code of the checkout as it is when they start**, and import
  some modules late. Do not switch branches in that checkout while jobs are
  queued; run other code from a git worktree with
  `PYTHONPATH=<worktree>/src`.
- **The progress bar is shown only on a terminal**. Follow a running job in
  the MLflow metrics (one row per epoch).
- **SIGTERM** (cancellation, time limit) ends the run cleanly: the MLflow run
  is not left `RUNNING`.
- **Jobs that create a new experiment at the same time** race in the folder
  store; they all join the oldest experiment of that name.
- **An evaluation logged into another experiment than the model's** leaves a
  metrics-only model folder there, which breaks MLflow's lookup of the model
  by `models:/<id>`. Refer to models by their training run ID: the trainer
  then loads them from their own artifact location.
