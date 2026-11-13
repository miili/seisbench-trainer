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
  The default was then made adaptive (see the next section).
- `val_loss` is not comparable between runs whose loss differs (weights,
  mask), so compare and select by `det/f1_mean`, now the default
  `training.monitor`.

Open: confirm N 0.5 against N 1 with several seeds and when fine-tuning.

## 2026-11-12: run length without a fixed epoch count

Same experiment, code `d09411a`, the new defaults otherwise (N 0.5,
`log_softmax`), at most 100 epochs, `min_delta` 0.002 for both the plateau
scheduler (factor 0.5) and early stopping. Best-epoch scores:

| run | epochs run | det/f1_mean | pick MAE P / S | LR drops at epoch | time |
| --- | --- | --- | --- | --- | --- |
| scratch, plateau 3, stop 10 | 39 | 0.844 | 0.060 / 0.062 s | 20, 25, 29, 34, 38 | 25 min |
| scratch, plateau 5, stop 15 | 62 | 0.846 | 0.058 / 0.060 s | 33, 40, 47, 54, 60 | 39 min |
| scratch, plateau 5, stop 10 | 34 | 0.842 | 0.063 / 0.063 s | 24, 31 | 24 min |
| scratch, cosine 40 epochs (round 2) | 40 | 0.845 | 0.062 / 0.063 s | - | 27 min |
| fine-tune, plateau 5, stop 10 | 42 | 0.847 | 0.060 / 0.060 s | 29, 39 | 28 min |
| fine-tune, cosine 40 epochs (round 1) | 40 | 0.846 | 0.061 / 0.061 s | - | 27 min |

- Plateau 5 with stop 10 ends a scratch run too early: both count the same
  flat epochs, so the run stopped 3 epochs after the second rate drop
  (rate still 0.00075). Fine-tuning, which starts closer to the optimum,
  was not affected.
- Plateau 3 with stop 10 matches cosine without an epoch count: five
  halvings, then the stop. Stop 15 was best on every metric, but within
  the noise of plateau 3, at 1.6 times the time.
- Default: plateau patience 5, early stopping 15, `min_delta` 0.002,
  `epochs` 100 as the limit: the best run, chosen for accuracy over time
  (about 40 min per BohemiaSaxony run). Plateau 3 with stop 10 is the
  faster option at about the same F1.

## 2026-11-13: fine-tuning volpick on deeplowfrequency

MLflow experiment `deeplowfrequency/PhaseNet-e2e` (two experiments of
that name: the jobs raced to create it, fixed in `set_experiment`); code
`7f12a62`; configs and `summary.py` in `~/sbt-runs/deeplowfrequency/`.
Dataset: local `deeplowfrequency` (129,637 traces, 100 Hz, ZNE; 99.7k
deep low-frequency event traces of about 120 s with the picks near 60 s,
29.9k noise traces of 60 s; seisbench splits 90.5k / 13.3k / 25.9k).
Fine-tuning `volpick` (ZNE, trained with lr 0.0005) with volpick as the
baseline, every other setting the code default (Adam lr 0.001, N 0.5,
sigma 0.1 s, plateau 5, stop 15, at most 100 epochs), one seed (42).
12-23 min per run on one A40, about 20 s per epoch at 100 Hz.

| run | epochs | det/f1_mean | F1 P / S | P thr | pick MAE P / S |
| --- | --- | --- | --- | --- | --- |
| volpick, no fine-tuning (100 Hz) | - | 0.807 | | | |
| defaults | 36 | 0.829 | 0.854 / 0.805 | 0.38 | 0.145 / 0.205 s |
| defaults, seed 43 | 43 | 0.829 | 0.855 / 0.804 | 0.38 | 0.139 / 0.205 s |
| lr 0.0005 | 33 | 0.828 | 0.855 / 0.802 | 0.38 | 0.141 / 0.204 s |
| lr 0.01 | 45 | 0.823 | 0.850 / 0.796 | 0.40 | 0.147 / 0.211 s |
| N weight 1 | 33 | 0.829 | 0.853 / 0.804 | 0.25 | 0.144 / 0.204 s |
| label sigma 0.2 s | 32 | **0.833** | 0.857 / 0.809 | 0.50 | 0.147 / 0.205 s |
| `real_noise` p 0.5 | 44 | 0.824 | 0.849 / 0.798 | 0.32 | 0.149 / 0.209 s |
| from scratch, lr 0.003 | 72 | 0.813 | 0.836 / 0.790 | 0.35 | 0.160 / 0.219 s |
| volpick, no fine-tuning (50 Hz) | - | 0.712 | | | |
| 50 Hz, sigma 0.1 s (5 samples) | 45 | 0.802 | 0.834 / 0.770 | 0.40 | 0.149 / 0.212 s |
| 50 Hz, sigma 0.2 s (10 samples) | 29 | 0.808 | 0.837 / 0.778 | 0.50 | 0.154 / 0.215 s |

- **The defaults hold.** Fine-tuning gains 0.023 over volpick and 0.017
  over training from scratch. lr 0.001 and volpick's own 0.0005 tie;
  0.01, our BohemiaSaxony fine-tuning rate, is 0.006 worse. N 0.5 ties
  with N 1 on F1 and again lifts the best threshold (0.38 against 0.25).
  The second seed matched the first within 0.0001.
- **Label sigma 0.2 s** was the best run (+0.003, P precision 0.869
  against 0.852 at equal recall; thresholds about 0.5). The onsets of
  deep low-frequency events are emergent. One seed; confirm before it
  becomes this dataset's setting. Not a global default.
- **`real_noise`** (p 0.5, scale 0-1, horizontals rotated, pick windows
  only) lowered validation F1 by 0.006. The validation windows carry no
  added noise and the training split already has 21k real noise traces,
  so this test cannot show a gain; it would show on noisier target data.
- **50 Hz** (volpick time-stretched: 3001 samples are 60 s, so
  `pick_window` is 120 s with 60 s before). A sigma in seconds halves
  in samples at 50 Hz; keeping the 10 samples the 100 Hz model saw
  (0.2 s) gained 0.006 over 0.1 s. Validation windows are 60 s instead of
  30 s, so the F1 is not directly comparable with the 100 Hz rows.
  Decimating on access makes an epoch about 50% slower.
- **Data**: 275 traces store NaN samples (volpick_alaska 214,
  volpick_cascade 61, all events): 239 a missing component as one NaN
  channel, 36 all three channels NaN although `trace_completeness` is
  about 1. Before `7f12a62` one NaN made the loss NaN and the run
  exported a model with F1 0; non-finite samples are now set to zero and
  a NaN loss stops the run. The 36 all-NaN traces (27 train) are blank
  windows with pick labels; they should be removed from the dataset
  (list: `~/sbt-runs/deeplowfrequency/nonfinite_traces.csv`).

Open: sigma 0.2 s with more seeds and sigma 0.3-0.4 s (also at 50 Hz,
to separate width in seconds from width in samples); `real_noise`
evaluated on noisier data than its own validation split.

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
