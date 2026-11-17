# Getting started

## Install

=== "Laptop (uv)"

    The repository pins CPU-only torch wheels for development.

    ```bash
    git clone https://github.com/miili/seisbench-trainer
    cd seisbench-trainer
    uv sync
    uv run seisbench-trainer --help
    ```

=== "GPU cluster (conda)"

    `pip` installs the CUDA torch wheels from PyPI.

    ```bash
    conda create -n seisbench-trainer python=3.13 pip
    conda activate seisbench-trainer
    pip install -e .            # add '.[hpo]' for hyperparameter search
    ```

Python 3.13 is required.

## A first run

The synthetic dataset needs no download and trains in about a minute on a
CPU:

```bash
uv run seisbench-trainer train configs/synthetic-smoke.json
uv run mlflow ui          # http://localhost:5000, reads ./mlflow.db
```

The run prints its ID and the validation scores of the best epoch. In the
MLflow UI it has the full config, the per-epoch metrics, the validation
figures (`evaluation/validation_P.html`) and the exported model.

## Your own run

Write the defaults to a file and edit what differs:

```bash
seisbench-trainer config > run.json
```

Every field has a default, so a config only needs what you change. For
example, PhaseNet fine-tuned from the pretrained `original` weights on
BohemiaSaxony:

```json
{
  "data": {
    "datasets": [{"type": "seisbench", "name": "BohemiaSaxony"}],
    "component_order": "ENZ"
  },
  "model": {"type": "PhaseNet", "pretrained": "original"}
}
```

Pretrained weights fix the component order and the sampling rate: `original`
was trained on `ENZ` at 100 Hz, and the trainer refuses a mismatch instead of
feeding swapped channels. Check a config before you start it; this prints
every resolved value, or the list of problems with exit code 2:

```bash
seisbench-trainer config run.json
```

Then train, and evaluate the best model on the test split:

```bash
seisbench-trainer train run.json
seisbench-trainer evaluate --model <run_id>
```

## Where results go

MLflow reads `MLFLOW_TRACKING_URI`. Without it, runs go to `./mlflow.db`
(SQLite). On a cluster without an MLflow server, use a folder on the shared
filesystem (see [Cluster](guide/cluster.md)):

```bash
export MLFLOW_TRACKING_URI=file:///shared/mlruns
export MLFLOW_ALLOW_FILE_STORE=true
```

Lightning checkpoints stay local in `training.checkpoint_dir`; only the best
model is uploaded.
