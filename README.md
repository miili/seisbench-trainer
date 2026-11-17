# seisbench-trainer

Train, evaluate and tune [SeisBench](https://github.com/seisbench/seisbench)
phase pickers with PyTorch Lightning, with every run tracked in
[MLflow](https://mlflow.org). One JSON file describes a run; the defaults are
the best settings found in our experiments.

**Documentation: <https://miili.github.io/seisbench-trainer/>**

- Trains PhaseNet from scratch or fine-tunes pretrained weights on any
  `seisbench.data` dataset, or several at once.
- Labels every catalogued event in a training window, not only the trace's
  own: in swarms this gives better pickers (+0.012 F1 on BohemiaSaxony, 17 %
  more located events in a qseek swarm benchmark).
- Evaluates every epoch against a pretrained baseline and on noise-added
  windows; selects thresholds on the dev split, never on test.
- Exports the best model as a SeisBench `.json`/`.pt` pair with its
  thresholds, ready for `seisbench.models.PhaseNet.load()` and qseek.
- Tunes hyperparameters with Optuna, with Slurm workers sharing one study.
- A CLI for people and agents: config checks with exit code 2, `--json`
  results on stdout, logs on stderr.

## Quick start

```bash
uv sync
uv run seisbench-trainer train configs/synthetic-smoke.json   # 1 min on a CPU
uv run mlflow ui                                              # reads ./mlflow.db
```

```bash
seisbench-trainer config > run.json          # every setting with its default
seisbench-trainer config run.json            # check a config, print what it resolves to
seisbench-trainer train run.json [--json]
seisbench-trainer evaluate --model <run_id> [--dataset ETHZ] [--json]
seisbench-trainer hpo run.json               # needs pip install '.[hpo]'
sbatch slurm/train.sbatch run.json           # on a Slurm cluster
```

On a GPU cluster: `pip install -e .` in a conda environment (Python 3.13),
and an MLflow folder store on the shared filesystem
(`MLFLOW_TRACKING_URI=file:///shared/mlruns`); see the
[cluster guide](https://miili.github.io/seisbench-trainer/guide/cluster/).

## Documentation

| | |
| --- | --- |
| [Getting started](docs/getting-started.md) | Install, a first run, your own config |
| [Guide](docs/guide/configuration.md) | Configuration, data, labels, training, evaluation, tracking, HPO, cluster, automation |
| [Config reference](docs/reference/config.md) | Every field, its default and meaning (generated) |
| [Findings](docs/findings.md) | What the experiments established; the [experiment log](docs/experiments.md) |
| [Development](docs/development.md) | Code layout, checks, docs; rules in [AGENTS.md](AGENTS.md) |

Preview the site locally with `uvx zensical serve`.
