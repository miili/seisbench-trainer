# Training

## Model and starting weights

`model` picks the picker and its starting weights:

```json
{"type": "PhaseNet", "pretrained": "volpick"}
```

`pretrained` is a SeisBench pretrained name (`original`, `volpick`,
`stead`, ...), the path of a local SeisBench model (its `.json` or `.pt`, or
the path without suffix), or `null` to train from scratch. On
deeplowfrequency, fine-tuning `volpick` gained 0.017 `det/f1_mean` over
training from scratch; on the much larger BohemiaSaxony, fine-tuning
`original` and training from scratch ended within the noise (0.847 and
0.846).

The model config is also the adapter between the trainer and the network:
it builds the model, its labels, its loss and its output mapping. Other
SeisBench pickers are added as new members of the `model` union, without
changing the rest of the code.

## Loss

`model.loss` is a soft-target cross-entropy over the softmax channels, taken
from the logits (`log_softmax`, as the original PhaseNet). The noise channel
weighs 0.5 by default: it balances P and S against the far more frequent
noise samples, gave the best F1 on BohemiaSaxony and lifts the F1-optimal
threshold from about 0.3 to 0.5. `{"from_logits": false, "weights": {}}` is
SeisBench's pick-benchmark loss.

## Run length and learning rate

By default a run finds its own length: the plateau scheduler halves the
learning rate after 5 epochs without improvement, early stopping ends the
run after 15, and `epochs` (100) is only the limit. An improvement must beat
the best value by `min_delta` (0.002, the run-to-run noise of
`det/f1_mean`).

| `training.scheduler` | |
| --- | --- |
| `plateau` (default) | Halve the rate when `training.monitor` stops improving |
| `cosine` | Anneal over all epochs; needs `early_stopping_patience: null` |
| `exponential` | Multiply the rate by `decay` every epoch (AI4EPS PhaseNet: 0.9) |
| `null` | Constant rate |

The optimizer is Adam with rate 0.001 (`adam` or `adamw`). For fine-tuning,
0.001 and 0.0005 tie and 0.01 is worse; the BohemiaSaxony runs from scratch
used 0.003.

## Model selection

`training.monitor` (default `det/f1_mean`, the mean of the P and S detection
F1 on the validation windows) picks the best epoch for checkpointing, early
stopping, the plateau scheduler and HPO. `val_loss` is also possible, but
not comparable between runs whose loss settings differ.

## Reproducibility

`training.seed` fixes the weights, the training augmentations and the
validation windows. GPU non-determinism leaves about 0.002 F1 between runs
of the same config and seed: compare settings with two seeds and treat
smaller differences as noise.

A run is reproducible from its MLflow record alone: it logs the resolved
config, the seed, the dataset names, the package versions and the
seisbench-trainer version, which names the git commit (setuptools-scm;
commit before you start runs).

## Precision and devices

One run uses one device (`accelerator`: `auto`, `cpu` or `gpu`).
`precision` can be `16-mixed` or `bf16-mixed`; the loss is computed from the
logits, which keeps it stable, but mixed precision has not been compared
against `32-true` on our data. A non-finite training loss stops the run
instead of exporting a broken model.
