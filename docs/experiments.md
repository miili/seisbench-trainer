# Experiments

Findings that set defaults in the code. Each entry names the runs, so the
numbers can be looked up in MLflow. All scores are on the validation split;
the test split is never used to choose settings.

## 2026-11-12: PhaseNet loss on BohemiaSaxony

MLflow experiment `BohemiaSaxony/PhaseNet-loss-grid`; code at `53dc2fc`
(round 1) and `0a1e4d1` (round 2). 40 epochs, cosine LR, Adam (lr 0.003
from scratch, 0.01 fine-tuning `original`), label sigma 0.1 s, one seed (42),
selection by `det/f1_mean`. About 27 min per run on one A30/A40
(34 s per epoch, BohemiaSaxony preloaded).

Best-epoch scores, from scratch:

| loss setting | det/f1_mean | P prec / rec | best P threshold | pick MAE P / S |
| --- | --- | --- | --- | --- |
| N weight 0.5 | **0.845** | 0.827 / 0.848 | 0.47 | 0.062 / 0.063 s |
| N weight 0.2 | 0.842 | 0.821 / 0.849 | 0.68 | 0.060 / 0.065 s |
| N weight 1 (`log_softmax`) | 0.841, 0.840 | 0.833 / 0.834 | 0.28-0.32 | 0.064-0.068 s |
| N weight 1, `log(softmax + eps)` | 0.840 | 0.836 / 0.826 | 0.35 | 0.068 / 0.064 s |
| loss mask 5 s around picks | 0.813 | 0.794 / 0.832 | 0.55 | 0.060 / 0.062 s |
| loss mask 2 s | 0.787 | 0.762 / 0.825 | 0.73 | 0.061 / 0.059 s |
| loss mask 0.5 s | 0.753 | 0.728 / 0.807 | 0.90 | 0.078 / 0.071 s |

Fine-tuning `original` (ENZ) behaved the same way and scored slightly
higher (0.846 without a mask, 0.757 with the 0.5 s mask). The pretrained
`original` PhaseNet without fine-tuning scores 0.696.

What we learned:

- **Run-to-run noise** with identical settings and seed is about 0.002 F1
  and 0.004 s pick MAE (GPU non-determinism). Smaller differences mean
  nothing; one seed per setting is only good for effects above about 0.005.
- **`from_logits`**: `log_softmax` and pick-benchmark's `log(softmax + eps)`
  train equally well (within 0.002). The default is `log_softmax`, which is
  numerically safer under mixed precision.
- **Noise weight**: down-weighting the noise channel to 0.5 gave the best
  F1 (+0.004 over N 1) and slightly better pick times. It also raises the
  peak heights, so the best threshold moves from about 0.3 to 0.5, which
  leaves more room for a fixed threshold at inference. N 0.2 is no better.
  Default: `model.loss.weights = {"N": 0.5}`.
- **Loss mask** (only samples near picks enter the loss in windows with
  picks; pick-free windows unmasked): worse at every width. Peaks get
  higher, but precision drops by 0.04-0.17 even at the best threshold: the
  model never learns that P and S coda are not picks, because almost every
  training window contains a pick and the coda beyond the mask is hidden.
  Pick timing was not better than with N 0.5. The feature was removed.
- **Training length**: improvements fell below the noise (0.002) after
  about 30 epochs, under cosine (40 epochs, gain in the last 10 epochs
  +0.001) and constant LR alike. Two 100-epoch fine-tuning runs with a
  constant rate of 0.01 (experiment `BohemiaSaxony/PhaseNet`, 2026-11-08)
  were within 0.005 of their best by epoch 18-21 and within 0.002 by
  epoch 31-45; after that F1 only drifted between 0.836 and 0.842. They
  ended at 0.843, below the 40-epoch cosine run (0.846): the decaying
  rate matters more than the length. One epoch is about 200k windows
  (198 steps of 1024), so other datasets need other epoch counts.
  Default since then: plateau scheduler (factor 0.5, patience 5) and
  early stopping (patience 10, `min_delta` 0.002), at most 100 epochs.
- `val_loss` is not comparable between runs whose loss differs (weights,
  mask), so compare and select by `det/f1_mean`, now the default
  `training.monitor`.

Open: confirm N 0.5 against N 1 with several seeds and when fine-tuning;
check the plateau + early-stopping default against the 40-epoch cosine
runs on BohemiaSaxony.

## Reference recipes

- PhaseNet (Zhu & Beroza 2019, doi:10.1093/gji/ggy423): softmax
  cross-entropy over noise, P and S, computed from the logits; labels are
  truncated Gaussians with sigma 0.1 s.
- AI4EPS PhaseNet code: Adam, lr 0.01, learning rate times 0.9 per epoch
  (staircase `exponential_decay`, one epoch of steps), no weight decay.
  Optional `class_weights` (off by default) scale each sample's whole
  cross-entropy by `sum_c w_c * y_c`; our `weights` act per channel inside
  the sum instead. It also offers IOU and MSE losses.
- SeisBench pick-benchmark: PhaseNet with `vector_cross_entropy`
  (`log(softmax + eps)`), label sigma 20 samples, 2/3 of the windows cut
  around a pick. EQTransformer with BCE on detection, P and S, weighted
  0.05 / 0.40 / 0.55. No class weights and no masking.
