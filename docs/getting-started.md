---
icon: lucide/download
---

# Installation

seisbench-trainer needs Python 3.13. It installs from the repository; the
[quickstart](quickstart.md) continues from here.

```bash
git clone https://github.com/miili/seisbench-trainer
cd seisbench-trainer
```

=== "Laptop (uv)"

    [uv](https://docs.astral.sh/uv/) installs the locked environment. It
    pins CPU-only torch wheels: right for development and smoke runs, too
    slow for real training.

    ```bash
    uv sync
    uv run seisbench-trainer --help
    ```

    Prefix every command with `uv run`, or activate `.venv`.

=== "GPU (pip or conda)"

    `pip` installs the CUDA torch wheels from PyPI. Use a virtual
    environment or a conda environment:

    ```bash
    conda create -n seisbench-trainer python=3.13 pip    # or: python3.13 -m venv .venv
    conda activate seisbench-trainer
    pip install -e .
    python -c "import torch; print(torch.cuda.is_available())"   # True
    ```

    An editable install (`-e`) keeps the version current with the git
    commit, which every run records.

## Optional: hyperparameter search

[Hyperparameter search](guide/hpo.md) needs Optuna, the `hpo` extra:

```bash
pip install -e '.[hpo]'        # with uv: uv sync (the dev group includes it)
```

## Check the installation

```bash
seisbench-trainer version
seisbench-trainer train configs/synthetic-smoke.json     # about a minute on a CPU
```

The smoke run generates its data and downloads only the small weights of the
pretrained baseline. Its results go to `./mlflow.db`; `mlflow ui` shows them.
Where to keep results on a cluster is in
[Tracking and models](guide/tracking.md#backends).

## What a run needs

- **Memory**: every split is held in memory. Plan RAM for the decoded
  waveforms at the training sampling rate.
- **A GPU** for real training; one run uses one device.
- **Disk** for the local Lightning checkpoints (`training.checkpoint_dir`)
  and the MLflow store.
