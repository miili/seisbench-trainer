---
icon: lucide/rocket
---

# Quickstart

From nothing to a trained, evaluated and exported picker in five steps. Every
step is one command; the defaults are the best settings we found, so a config
only names your data.

## 1. Install

```bash
git clone https://github.com/miili/seisbench-trainer
cd seisbench-trainer
uv sync                                   # CPU torch; on a GPU machine: pip install -e .
```

## 2. A first run, in a minute

The synthetic dataset needs no download:

```bash
uv run seisbench-trainer train configs/synthetic-smoke.json
```

It trains PhaseNet for three epochs on a CPU, evaluates every epoch and
exports the best model. The last lines show the run and its scores:

```text
Run <run_id> (sqlite:///mlflow.db)
best_det/f1_mean = …
┏━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ model   ┃ phase ┃    F1 ┃ prec. ┃ recall ┃  thr. ┃ MAE s ┃   AUC ┃
┡━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ trained │ P     │     … │     … │      … │     … │     … │     … │
```

Three epochs on synthetic data only show that everything works; the scores
mean little.

Look at it in MLflow, with the per-epoch metrics and the validation figures:

```bash
uv run mlflow ui                          # http://localhost:5000
```

## 3. Your data

Name a dataset and, to fine-tune, pretrained weights. Everything else has a
default:

```json title="run.json"
{
  "data": {
    "datasets": [{"type": "seisbench", "name": "ETHZ"}],
    "component_order": "ENZ"
  },
  "model": {"type": "PhaseNet", "pretrained": "original"}
}
```

A SeisBench dataset is downloaded on first use; a local one is
`{"type": "local", "path": "~/data/my-dataset"}`. Pretrained weights fix the
component order (`original` is `ENZ`). Check the config, then train:

```bash
uv run seisbench-trainer config run.json  # every resolved value, or the problems
uv run seisbench-trainer train run.json
```

!!! tip "On a GPU cluster"
    `sbatch slurm/train.sbatch run.json` runs the same on one GPU, with
    MLflow in a folder on the shared filesystem. See
    [Slurm cluster](guide/cluster.md).

## 4. Evaluate

Thresholds are selected on the dev split and applied to the test split:

```bash
uv run seisbench-trainer evaluate --model <run_id>
uv run seisbench-trainer evaluate --model <run_id> --dataset BohemiaSaxony   # another dataset
```

Add `--json` to any command for a machine-readable result.

## 5. Use the model

The best model is a SeisBench `.json`/`.pt` pair with its validation
thresholds. Download it from MLflow and load it like any SeisBench model:

```bash
uv run mlflow artifacts download -u models:/<model_id> -d model
```

```python
import seisbench.models as sbm

model = sbm.PhaseNet.load("model/extra_files/PhaseNet-ETHZ-from-original")
picks = model.classify(stream).picks      # uses the model's own thresholds
```

qseek takes the same pair as its `pretrained` image function model.

## Next

- [Configuration](guide/configuration.md): how a config is built, and every
  field in the [config reference](reference/config.md).
- [Labels](guide/labels.md): why every event in a window is labelled.
- [Scripts and agents](guide/automation.md): JSON results and exit codes.
- [Findings](findings/index.md): what the defaults are based on.
