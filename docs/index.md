# seisbench-trainer

Train, evaluate and tune [SeisBench](https://github.com/seisbench/seisbench)
phase pickers with PyTorch Lightning, with every run tracked in
[MLflow](https://mlflow.org). One JSON file describes a run; the defaults are
the best settings found in [our experiments](findings.md).

```bash
seisbench-trainer config > run.json      # every setting, with its default
seisbench-trainer config run.json        # check a config, show what it resolves to
seisbench-trainer train run.json         # train, evaluate per epoch, export the best model
seisbench-trainer evaluate --model <run_id>
```

## What it does

- **Trains PhaseNet** from scratch or fine-tunes pretrained weights (SeisBench
  names such as `original` or `volpick`, or a local model) on one or several
  `seisbench.data` datasets, preloaded into memory.
- **Labels every event in a window.** A SeisBench metadata row holds only its
  own event. In swarms, other events' arrivals in the same window would be
  labelled noise; the trainer finds them by station and time and labels them
  too ([Labels](guide/labels.md)).
- **Evaluates every epoch** with pick-accuracy and detection metrics, against
  a pretrained baseline, on clean and on noise-added validation windows; the
  test split is never used for selection.
- **Exports the best model** as a SeisBench `.json`/`.pt` pair, with its
  validation thresholds, ready for `seisbench.models.PhaseNet.load()` and
  qseek.
- **Tracks everything in MLflow**: config, versions, metrics, figures, the
  model. A run is reproducible from its record.
- **Tunes hyperparameters** with Optuna, with any number of Slurm workers
  sharing one study.

## Where to go

| | |
| --- | --- |
| New here | [Getting started](getting-started.md) |
| Writing a config | [Configuration](guide/configuration.md), [Config reference](reference/config.md) |
| Scripts and agents | [Automation](guide/automation.md), [CLI reference](reference/cli.md) |
| Why the defaults are what they are | [Findings](findings.md), [Experiment log](experiments.md) |
| Changing the code | [Development](development.md) |
