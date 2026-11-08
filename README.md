# seisbench-trainer

Train, evaluate and optimise [SeisBench](https://github.com/seisbench/seisbench)
phase pickers with PyTorch Lightning. Every run is tracked in
[MLflow](https://mlflow.org) 3. A run is described by one JSON config file.

Status: early development. See [docs/plan.md](docs/plan.md) for what is
implemented and [docs/requirements.md](docs/requirements.md) for the goals.

## Laptop (development)

```bash
uv sync                                  # CPU-only torch
uv run pre-commit install
uv run pytest

uv run seisbench-trainer config > run.json            # full default config
uv run seisbench-trainer train configs/synthetic-smoke.json
uv run mlflow ui                                      # reads ./mlflow.db
```

`configs/synthetic-smoke.json` uses the built-in synthetic dataset, so
nothing is downloaded. It trains in under a minute on a CPU.

## Slurm cluster

Once, create the conda environment:

```bash
conda create -n seisbench-trainer python=3.13 pip
conda activate seisbench-trainer
pip install -e .                          # CUDA torch from PyPI
```

Each run:

```bash
export MLFLOW_TRACKING_URI=file:///<shared>/mlruns
sbatch slurm/train.sbatch configs/bohemiasaxony-phasenet-original.json
```

MLflow stores the runs in a folder on the shared filesystem; no server is
needed. The Slurm script sets `MLFLOW_ALLOW_FILE_STORE=true`, which MLflow
3.18 requires for this. To browse the results, run
`mlflow ui --backend-store-uri file:///<shared>/mlruns` on a login node and
open it through an SSH tunnel.

## Config

`seisbench-trainer config --schema` prints the JSON schema. Choices between
alternatives are tagged with `"type"`, for example:

```json
{
  "data": {"datasets": [{"type": "seisbench", "name": "BohemiaSaxony"}], "component_order": "ENZ"},
  "model": {"type": "PhaseNet", "pretrained": "original"},
  "training": {"optimizer": {"type": "adam", "lr": 0.01}}
}
```

Fine-tuning a pretrained model requires its own component order and sampling
rate. For example, `original` needs `ENZ`.

## Outputs per run

- metrics per epoch and system metrics;
- `config.json`, flattened params, and tags (git commit, package versions,
  host, Slurm IDs);
- the best model as an MLflow logged model, with the SeisBench `.json`/`.pt`
  pair under `extra_files/`; it loads with `seisbench.models.PhaseNet.load()`
  and can be used directly in Qseek.

Lightning checkpoints stay local in `training.checkpoint_dir`.
