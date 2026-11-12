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

## Evaluate

```bash
seisbench-trainer evaluate --model <run_id>                     # its test split
seisbench-trainer evaluate --model models:/<name>@champion --dataset ETHZ
seisbench-trainer evaluate --model PhaseNet:original --dataset BohemiaSaxony
sbatch slurm/evaluate.sbatch --model <run_id>                   # on the cluster
```

Thresholds are selected on the dev split and applied to the evaluated split
at 0.1, 0.2 and 0.5 s tolerance. Results are linked to the model and the
dataset in MLflow. Set `"evaluation": {"test_at_end": true}` to evaluate
right after training.

## Hyperparameter search

Needs the optional extra: `pip install -e '.[hpo]'`. Add an `hpo` section
to a run config (see `configs/bohemiasaxony-phasenet-hpo.json`): the study
name, the journal file on the shared filesystem, the number of trials, the
sampler, the pruner, and the parameters as dotted config paths.

```bash
uv run seisbench-trainer hpo configs/synthetic-hpo.json          # laptop
sbatch --array=0-3 slurm/hpo_worker.sbatch configs/bohemiasaxony-phasenet-hpo.json
```

All workers join one Optuna study. In MLflow the study is a parent run
(`hpo-<study>`, with the best trial and the Optuna plots) and each trial is
a child run.

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
- `config.json`, flattened params, and tags (package versions; the seisbench-trainer
  version names the git commit,
  host, Slurm IDs);
- `evaluation/validation_P.html` and `validation_S.html`: the validation
  figures of all epochs with a slider (open them in the Artifacts tab);
- the best model as an MLflow logged model named
  `<model>-<datasets>-<variant>`, with the SeisBench `.json`/`.pt`
  pair under `extra_files/`, with the validation thresholds in its
  `default_args` (`P_threshold`, `S_threshold`) and the best epoch's
  figures next to it; it loads with `seisbench.models.PhaseNet.load()`
  and can be used directly in Qseek.

Lightning checkpoints stay local in `training.checkpoint_dir`.
