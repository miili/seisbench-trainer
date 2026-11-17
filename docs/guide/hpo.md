# Hyperparameter search

Needs the optional extra: `pip install 'seisbench-trainer[hpo]'` (Optuna).

Add an `hpo` section to a run config. Parameters are dotted paths into the
config; every trial applies its values to a copy of the config, validates it
and trains it like any run:

```json
{
  "hpo": {
    "study_name": "bohemiasaxony-lr",
    "storage": "/shared/optuna/bohemiasaxony-lr.journal",
    "n_trials": 40,
    "sampler": {"type": "tpe", "n_startup_trials": 10, "seed": 42},
    "pruner": {"type": "median", "n_startup_trials": 5, "n_warmup_steps": 5},
    "params": [
      {"type": "float", "path": "training.optimizer.lr", "low": 1e-4, "high": 3e-2, "log": true},
      {"type": "categorical", "path": "training.optimizer.type", "choices": ["adam", "adamw"]},
      {"type": "float", "path": "augmentation.label_sigma_s", "low": 0.05, "high": 0.3}
    ]
  }
}
```

```bash
seisbench-trainer hpo run.json                                  # one worker
sbatch --array=0-3 slurm/hpo_worker.sbatch run.json             # four workers
```

- **Workers share one study** through an Optuna journal file on a shared
  filesystem; start as many as you like. They stop once `n_trials` trials
  have finished (each may start one more).
- **The data is loaded once per worker** and reused by its trials. Paths that
  would change the data or the objective cannot be tuned: `data.datasets`,
  `data.component_order`, `data.sampling_rate`, `augmentation.label_events`
  and `training.monitor`.
- **The objective** is the best `training.monitor` of the trial. Trials never
  touch the test split.
- **In MLflow** the study is a parent run (`hpo-<study>`, with
  `best_trial.json` and the Optuna plots) and every trial a child run.
- **Pruning** stops trials that fall behind on `training.monitor`; a pruned
  trial's run is marked `KILLED`.

Search-space paths are checked when the config loads: each parameter's
example value is applied and the config validated, so a wrong path fails
before any trial runs.
