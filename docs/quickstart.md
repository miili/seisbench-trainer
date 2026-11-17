---
icon: lucide/rocket
---

# Quickstart

From nothing to a trained, evaluated and exported picker in five steps.
The defaults are the best settings we found, so a config only names your
data. Commands are shown with `uv run`; in a pip or conda environment, drop
that prefix.

## 1. Install

=== "Laptop (CPU)"

    ```bash
    git clone https://github.com/miili/seisbench-trainer
    cd seisbench-trainer
    uv sync                 # pins CPU-only torch: right for a laptop, slow for real training
    ```

=== "GPU machine"

    ```bash
    git clone https://github.com/miili/seisbench-trainer
    cd seisbench-trainer
    python3.13 -m venv .venv && source .venv/bin/activate    # or a conda env
    pip install -e .        # CUDA torch from PyPI
    python -c "import torch; print(torch.cuda.is_available())"   # should print True
    ```

More options (conda, the `hpo` extra, MLflow backends) are in
[Installation](getting-started.md).

## 2. A first run, in a minute

The synthetic dataset is generated on the fly; only the small weights of the
pretrained baseline are downloaded:

```bash
uv run seisbench-trainer train configs/synthetic-smoke.json
```

It trains PhaseNet for three epochs on a CPU, evaluates every epoch and
exports the best model. The output ends with the run, the model and the
scores of the best epoch:

```text
Run <run_id> (sqlite:///mlflow.db)
Model m-<model_id> (PhaseNet-synthetic-scratch)
best_det/f1_mean = …
┏━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ model   ┃ phase ┃    F1 ┃ prec. ┃ recall ┃  thr. ┃ MAE s ┃   AUC ┃
┡━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ trained │ P     │     … │     … │      … │     … │     … │     … │
```

Three epochs on synthetic data only show that everything works; the scores
mean little. Browse the run, its metrics per epoch and its figures:

```bash
uv run mlflow ui                          # http://localhost:5000
```

## 3. Your data

A config names what differs from the defaults. `{}` is a valid config (it
trains PhaseNet from scratch on ETHZ); this one fine-tunes the pretrained
`original` PhaseNet instead, as in `configs/ethz-phasenet-original.json`:

```json title="run.json"
{
  "data": {
    "datasets": [{"type": "seisbench", "name": "ETHZ"}],
    "component_order": "ENZ"
  },
  "model": {"type": "PhaseNet", "pretrained": "original"}
}
```

- **Datasets**: any `seisbench.data` class by name (downloaded on first use),
  your own SeisBench-format folder (`{"type": "local", "path":
  "~/data/my-dataset"}`, see [Bring your own dataset](guide/data.md#bring-your-own-dataset)),
  or several at once.
- **Memory**: every split is held in memory while training. Plan RAM for the
  decoded waveforms: a few GB for ETHZ, more than 100 GB for the largest
  SeisBench datasets.
- **Component order**: pretrained weights fix it (`original` is `ENZ`), and
  the trainer refuses a mismatch. The baseline model of the evaluation is
  fed in its own order automatically.

Check the config, then train. On a CPU this takes hours; use a GPU for real
runs:

```bash
uv run seisbench-trainer config run.json  # every resolved value, or the list of problems
uv run seisbench-trainer train run.json
```

!!! tip "On a GPU cluster"
    Point MLflow at a folder on the shared filesystem and submit the same
    config with the included Slurm script (see [Slurm cluster](guide/cluster.md)):

    ```bash
    export MLFLOW_TRACKING_URI=file:///<shared>/mlruns
    sbatch slurm/train.sbatch run.json
    ```

## 4. Evaluate

Detection thresholds are selected on the validation (dev) split and applied
to the test split, at 0.1, 0.2 and 0.5 s tolerance:

```bash
uv run seisbench-trainer evaluate --model <run_id>                    # its own test split
uv run seisbench-trainer evaluate --model <run_id> --dataset Iquique  # another dataset
```

Add `--json` to any command for a machine-readable result.

## 5. Use the model

The best model is a SeisBench `.json`/`.pt` pair named
`<model>-<datasets>-<variant>` (here `PhaseNet-ETHZ-from-original`), with
its validation thresholds. Download it by the model ID that `train` printed
and load it like any SeisBench model:

```bash
uv run mlflow artifacts download -u models:/<model_id> -d model
```

```python
import seisbench.models as sbm

model = sbm.PhaseNet.load("model/extra_files/PhaseNet-ETHZ-from-original")
picks = model.classify(stream).picks      # uses the model's own thresholds
```

Tools built on SeisBench, such as qseek, take the same pair.

## What a run produces

| Where | What |
| --- | --- |
| MLflow run | The resolved `config.json`, parameters, package versions, metrics per epoch, system metrics |
| `evaluation/validation_P.html`, `_S.html` | The validation figures of every epoch, with a slider |
| MLflow logged model, `extra_files/` | The SeisBench `.json`/`.pt` pair, `train_config.json`, the best epoch's figures |
| `checkpoints/<run_id>/` (local) | The Lightning checkpoint of the best epoch |

## Next

- [Configuration](guide/configuration.md): how a config is built, and every
  field in the [config reference](reference/config.md).
- [Data](guide/data.md): sources, splits, preprocessing and your own datasets.
- [Labels](guide/labels.md): why every event in a window is labelled.
- [Scripts and agents](guide/automation.md): JSON results and exit codes.
- [Findings](findings/index.md): what the defaults are based on.
