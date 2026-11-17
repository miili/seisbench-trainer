---
icon: lucide/package
---

# Tracking and models

## What a run logs

Every training run is one MLflow run in the experiment `tracking.experiment`
(default `<datasets>/<model>`, e.g. `BohemiaSaxony/PhaseNet`):

- **Parameters and tags**: the resolved config (`config.json` and flattened
  params), the versions of seisbench-trainer, SeisBench, torch, Lightning and
  MLflow, the host and the Slurm job IDs, and `tracking.tags`.
- **Metrics** per epoch ([Evaluation](evaluation.md)), the learning rate, and
  system metrics (CPU, GPU, memory).
- **Figures**: `evaluation/validation_<phase>.html`, all epochs with a slider.
- **The best model** as an MLflow logged model (below).

The seisbench-trainer version comes from git (setuptools-scm), so it names
the commit the run used. Commit before you start runs: a dirty tree gives a
`.dYYYYMMDD` version that does not identify the code.

## The exported model

The best epoch is logged as an MLflow model named
`<model>-<datasets>-<variant>`, e.g. `PhaseNet-BohemiaSaxony-from-original`.
Under `extra_files/` it carries:

- the SeisBench pair `<name>.json` / `<name>.pt`, which load with
  `seisbench.models.PhaseNet.load("<name>")`, with the component order, the
  sampling rate and the validation-optimal thresholds (`P_threshold`,
  `S_threshold` in `default_args`);
- `train_config.json`, the resolved run config;
- the validation figures of the best epoch.

Load a model for evaluation by run ID, `models:/<model_id>` or a registered
name; `seisbench-trainer evaluate` downloads it and reuses its training config.

## Using a model in qseek

Copy the SeisBench pair out of the MLflow store and point qseek's image
function at it:

```json
"image_function": {
  "image": "SeisBench",
  "model": "PhaseNet",
  "pretrained": "models/phasenet-bohemia.json"
}
```

qseek uses its own picker thresholds, not the model's. Older SeisBench
versions reject a model saved by a newer one; lower `seisbench_requirement`
in the `.json` if the qseek environment is older. For a model trained at
50 Hz, let qseek resample to 50 Hz (`pre_processing` `resample`), not
SeisBench.

## Backends

MLflow reads `MLFLOW_TRACKING_URI`; the code sets nothing else. Locally the
default is `sqlite:///mlflow.db`. On a cluster without a server, use a folder
on the shared filesystem (`file:///shared/mlruns` with
`MLFLOW_ALLOW_FILE_STORE=true`); never SQLite on a network filesystem.
