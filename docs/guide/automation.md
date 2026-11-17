# Scripts and agents

The CLI is built to be driven by scripts and coding agents as well as by
people: every input is checked before anything heavy runs, results and logs
are separate streams, and every command can answer in JSON.

## Contract

| | |
| --- | --- |
| stdout | Results only: a table, or with `--json` one JSON object |
| stderr | Logs, and everything a library prints during a run |
| exit 0 | Success |
| exit 1 | A failed run (an exception, a missing optional extra) |
| exit 2 | Invalid input: a bad argument, a missing file, an invalid config |

Progress bars are shown only when stderr is a terminal.

## Check before you run

```bash
seisbench-trainer config run.json > resolved.json || exit 1
```

validates the config without loading data or models and prints it with every
default filled in. An invalid config lists every problem by its path, one
line each:

```text
run.json: invalid run config
  augmentation.train.0: Input tag 'cut' found using 'type' does not match any of the expected tags: 'pick_window', 'random_window', 'channel_dropout', 'add_gap', 'gaussian_noise', 'rotate_horizontal', 'real_noise'
  training.epochs: Input should be greater than 0
```

`seisbench-trainer config --schema` prints the JSON schema with a
description of every field: give it to an agent that writes configs.

## JSON results

```bash
seisbench-trainer train run.json --json > result.json
```

```json
{
  "run_id": "4a0de5358e4b4e30a0fe11e0869b046b",
  "experiment_id": "820209881396864699",
  "tracking_uri": "file:///shared/mlruns",
  "model_id": "m-9c82819e50174395915292eec9f1e1ee",
  "model_name": "PhaseNet-BohemiaSaxony-scratch",
  "monitor": "det/f1_mean",
  "best": 0.857,
  "metrics": {"det/f1_mean": 0.857, "det_P/f1": 0.861, "...": 0.0}
}
```

`evaluate --json` returns the run ID, the model and one row per phase and
tolerance (threshold, precision, recall, F1, residuals, counts); `hpo --json`
returns the trials, best first. Non-finite values are `null`.

## A loop an agent can run

1. Write or change a config; check it with `seisbench-trainer config`.
2. Train it, e.g. as a Slurm job: `sbatch slurm/train.sbatch run.json`.
3. Read the result from MLflow (`mlflow.get_run(run_id).data.metrics`) or
   from `train --json`.
4. Compare settings with two seeds; differences below 0.002 `det/f1_mean`
   are noise.
5. Record findings that change a default in `docs/experiments.md`, with the
   MLflow experiment and the commit.
