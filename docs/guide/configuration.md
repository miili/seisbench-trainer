---
icon: lucide/sliders-horizontal
---

# Configuration

A run is described by one JSON file, validated by
[Pydantic](https://docs.pydantic.dev). There is no config layering: the
file plus the defaults is the run, and the resolved config is logged with
every run (`config.json` in MLflow).

## Sections

| Section | Contents |
| --- | --- |
| `data` | Datasets, component order, sampling rate, batch size, workers |
| `model` | Picker model, starting weights, loss |
| `augmentation` | Training windows and perturbations, normalization, labels |
| `training` | Epochs, seed, device, optimizer, scheduler, model selection |
| `evaluation` | Metric settings, baseline model, noisy validation, figures |
| `tracking` | MLflow experiment, run name, tags |
| `hpo` | Hyperparameter search (only for `seisbench-trainer hpo`) |

Every field, its type, default and meaning is in the
[config reference](../reference/config.md), generated from the code. The
same descriptions are in the JSON schema:

```bash
seisbench-trainer config --schema
```

## Choices are tagged with `type`

Wherever there are alternatives (a dataset source, a training step, an
optimizer, a scheduler, a noise level, an HPO sampler), the value is an object
whose `type` names the alternative. Unknown keys are errors, so a typo cannot
pass silently:

```json
{
  "training": {
    "optimizer": {"type": "adamw", "lr": 0.0005, "weight_decay": 0.01},
    "scheduler": {"type": "cosine"},
    "early_stopping_patience": null
  }
}
```

## Units

Durations are in seconds (`label_sigma_s`, `tolerance_s`, `window_s`, ...),
rates in Hz. The trainer converts seconds to samples once, with
`data.sampling_rate`, so a config means the same at any sampling rate.

## Checking a config

```bash
seisbench-trainer config run.json
```

prints the config with every default filled in, or lists the problems by
path and exits with code 2:

```text
run.json: invalid run config
  training.epochs: Input should be greater than 0
  training.lr: Extra inputs are not permitted
```

Some checks need the data or the model and run when training starts: a
pretrained model's component order and sampling rate must match `data`, and a
`pick_window` must be longer than the model input.

## Example configs

The repository's `configs/` folder holds working examples; the tests validate
all of them:

| File | Run |
| --- | --- |
| `synthetic-smoke.json` | Laptop smoke run on generated data |
| `synthetic-hpo.json` | A small HPO study on generated data |
| `ethz-phasenet-original.json` | Fine-tuning `original` on the public ETHZ dataset (the quickstart) |
| `bohemiasaxony-phasenet-scratch.json` | PhaseNet from scratch on BohemiaSaxony |
| `bohemiasaxony-phasenet-original.json` | Fine-tuning `original` on BohemiaSaxony |
| `bohemiasaxony-phasenet-hpo.json` | HPO on BohemiaSaxony |
| `deeplowfrequency-phasenet-volpick.json` | Fine-tuning `volpick` on a local dataset |
| `deeplowfrequency-phasenet-volpick-50hz.json` | The same at 50 Hz with 60 s windows |
