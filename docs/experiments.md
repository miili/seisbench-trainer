---
icon: lucide/flask-conical
---

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
`7f12a62`.
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
  windows with pick labels; they should be removed from the dataset.

After this suite the defaults became: `normalization` std (for PhaseNet
a no-op: `annotate_batch_pre` normalizes each channel again by the
model's own `norm`, as at inference) and the train steps `pick_window`,
`random_window`, `rotate_horizontal` 0.5, `real_noise` 0.5 (a no-op
with a warning on datasets without noise traces). Real noise is a
default for robustness on noisy target data, which this validation split
cannot measure; experiment `deeplowfrequency/PhaseNet-e2e-v2` reruns the
suite on these defaults with the augmentations ablated.

### The suite on the new defaults

MLflow experiment `deeplowfrequency/PhaseNet-e2e-v2`, code `dbba5fd`
(one experiment: the race fix held for 12 jobs started together). Best
epochs:

| run | epochs | det/f1_mean | F1 P / S | P thr | pick MAE P / S |
| --- | --- | --- | --- | --- | --- |
| defaults (rotate + real noise) | 24 | 0.820 | 0.847 / 0.793 | 0.35 | 0.149 / 0.209 s |
| defaults, seed 43 | 46 | 0.826 | 0.850 / 0.802 | 0.38 | 0.148 / 0.206 s |
| rotation only (no real noise) | 36 | 0.829 | 0.854 / 0.804 | 0.38 | 0.142 / 0.206 s |
| real noise only (no rotation) | 44 | 0.824 | 0.851 / 0.798 | 0.35 | 0.148 / 0.209 s |
| neither (the first suite's defaults) | 38 | **0.830** | 0.855 / 0.805 | 0.40 | 0.144 / 0.206 s |
| lr 0.0005 | 35 | 0.823 | 0.849 / 0.797 | 0.38 | 0.147 / 0.209 s |
| lr 0.01 | 44 | 0.821 | 0.845 / 0.797 | 0.38 | 0.148 / 0.210 s |
| N weight 1 | 36 | 0.823 | 0.849 / 0.796 | 0.22 | 0.146 / 0.207 s |
| label sigma 0.2 s | 34 | 0.829 | 0.853 / 0.805 | 0.47 | 0.150 / 0.210 s |
| from scratch, lr 0.003 | 71 | 0.809 | 0.835 / 0.783 | 0.38 | 0.166 / 0.225 s |
| 50 Hz, sigma 0.1 s | 48 | 0.798 | 0.830 / 0.767 | 0.38 | 0.151 / 0.213 s |
| 50 Hz, sigma 0.2 s | 38 | 0.806 | 0.839 / 0.773 | 0.47 | 0.156 / 0.217 s |

- Reproducible across the code changes: without the augmentations the
  run scored 0.830, the first suite's defaults 0.829.
- Rotation is neutral (0.829 against 0.830). Real noise costs about
  0.006 validation F1 (defaults 0.820 and 0.826 over two seeds, real
  noise alone 0.824), as in the first suite. The validation windows carry
  no added noise, so this measures only the cost; the benefit on noisier
  data is not measured yet.
- With real noise the epoch-to-epoch F1 is noisier. Seed 42 set its
  early-stopping reference at a high epoch 8 and stopped at epoch 24
  after two LR halvings, still improving (0.8199 at the last epoch);
  seed 43 ran 46 epochs. The two seeds differ by 0.006, three times the
  noise without augmentation.
- The other comparisons repeat the first suite under the new defaults:
  lr 0.001 and 0.0005 tie, 0.01 is no better; N 1 ties on F1 with a lower
  threshold (0.22); sigma 0.2 s is ahead (+0.006 over the defaults' mean)
  at 100 Hz and at 50 Hz (+0.007).

Real noise stays a default (it is expected to generalize and harden the
model). To measure that, every run now also validates on the dev windows
with real noise from the dev split's own noise traces added (noise peak
0.5-1 times the signal peak, seeded per trace; `noisy/det/f1_mean`).
`real_noise` gained a `level`: `peak` (ours, noise peak uniform 0-1 times
the window peak) or `std` (the dlf-catalog reference: noise std
log-uniform 0.1-2 times the window std over the non-zero channels).
Missing (all-zero) components get no noise in either. `channel_dropout`
0.2 joined the default steps, after the noise. Experiment
`deeplowfrequency/PhaseNet-noise-sigma` compares them.

### Real noise on noisy validation; label sigma in seconds or samples

MLflow experiment `deeplowfrequency/PhaseNet-noise-sigma`, code
`c4d470c`. Defaults
as above (rotate 0.5, real noise 0.5 `peak`, dropout 0.2, sigma 0.1 s).
`noisy` is `noisy/det/f1_mean` at the epoch the clean F1 selected; the
volpick baseline scores 0.807 clean and 0.585 noisy.

| run (seeds 42 / 43) | det/f1_mean | noisy | pick MAE P |
| --- | --- | --- | --- |
| real noise `peak` (default) | 0.822 / 0.825 | **0.663 / 0.666** | 0.149 / 0.146 s |
| real noise `std` (dlf-catalog) | 0.823 / 0.825 | 0.659 / 0.660 | 0.147 / 0.143 s |
| no real noise | 0.829 / 0.830 | 0.601 / 0.604 | 0.143 / 0.141 s |
| no dropout | 0.822 / 0.826 | 0.664 / 0.669 | 0.150 / 0.147 s |
| no augmentation (seed 42) | 0.828 | 0.596 | 0.143 s |

| label sigma | 100 Hz: F1, MAE P | 50 Hz: F1, MAE P |
| --- | --- | --- |
| 0.1 s | 0.824 (mean), 0.148 s (10 samples) | 0.796, 0.154 s (5 samples) |
| 0.2 s | 0.828, 0.152 s (20 samples) | 0.806, 0.156 s (10 samples) |
| 0.4 s | 0.832, 0.180 s (40 samples) | 0.809, 0.180 s (20 samples) |
| 0.8 s | 0.725, 0.328 s (80 samples) | 0.717, 0.315 s (40 samples) |

- **Real noise generalizes.** It gains 0.06 F1 on the noisy windows
  (0.601 to 0.665) for 0.005 on the clean ones. It stays a default.
- **`peak` and `std` scaling tie** on clean windows and are within 0.005
  on the noisy ones, which are themselves peak-scaled (a small bias
  towards `peak`). The default stays `peak`; `std` reproduces the
  dlf-catalog reference.
- **Channel dropout 0.2 is neutral** on both validations; neither has
  dropped components, where it should help. It stays a default.
- **Label sigma belongs in seconds.** The curves match in seconds at both
  rates and not in samples: 0.2 s gains over 0.1 s, 0.4 s adds little F1
  but costs 0.03 s pick MAE at both rates, and 0.8 s collapses at both
  (40 samples is the best F1 at 100 Hz and a collapse at 50 Hz). The 50 Hz
  gain of 0.2 s in the earlier suites is the better width in seconds, not
  volpick's label width in samples. For deep low-frequency events 0.2 s is
  the best trade-off (F1 +0.005, MAE unchanged).

Open: sigma 0.2 s with more seeds and sigma 0.3-0.4 s (also at 50 Hz,
to separate width in seconds from width in samples); `real_noise`
evaluated on noisier data than its own validation split.

## 2026-11-13: against the DLF v4b reference model

MLflow experiment `deeplowfrequency/PhaseNet-vs-v4b`, code `8f8cab1`.
The baseline of every run is v4b, the picker of the Laacher See DLF
catalogue: volpick fine-tuned on the same 21,983 events.

**Splits:** `deeplowfrequency` was split anew, not like v4b. v4b trained on
3,050 of our 4,397 test events (13,751 of 19,937 test traces) and on 1,564
of 2,199 dev events, so its scores here are optimistic. By decision, our
splits are the benchmark anyway.

**Partial labels:** v4b treats a missing P or S pick as "not read": the
loss is masked where it can arrive. Code `8f8cab1` added that as
`augmentation.missing_picks` (default `unread`) for the loss and the
metrics. Under `unread`, all P F1 values rise (v4b 0.83 to 0.88 on test),
because a correct pick of an unread P no longer counts as false.

Test F1 at 0.5 s tolerance with the dev thresholds, pick MAE; validation
`det/f1_mean` and `noisy/det/f1_mean` at the best epoch, against v4b on the
same validation windows (the windows depend on the seed):

| model | test F1 P / S | test MAE P / S | val (v4b) | noisy val (v4b) |
| --- | --- | --- | --- | --- |
| v4b (reference weights) | 0.876 / 0.786 | 0.146 / 0.210 s | | |
| volpick, no fine-tuning | 0.863 / 0.783 | 0.152 / 0.216 s | | |
| ours, sigma 0.2 s, seed 42 | 0.871 / **0.790** | 0.153 / 0.214 s | **0.842** (0.840) | **0.678** (0.663) |
| ours, sigma 0.2 s, seed 43 | 0.869 / 0.786 | 0.157 / 0.218 s | **0.843** (0.842) | **0.683** (0.661) |
| ours, sigma 0.2 s, `std` noise | 0.870 / 0.789 | 0.154 / 0.214 s | 0.842 (0.840) | 0.675 (0.663) |
| ours, sigma 0.1 s | 0.867 / 0.777 | 0.152 / 0.216 s | 0.834 (0.840) | 0.673 (0.664) |
| ours, sigma 0.2 s, trained `absent` | 0.867 / 0.794 | 0.155 / 0.213 s | | |
| v4b's recipe, trained here | 0.870 / 0.787 | 0.154 / 0.217 s | 0.840 (0.841) | 0.669 (0.665) |

("ours": rotate 0.5, real noise 0.5 `peak`, dropout 0.2, `unread`, Adam
0.001, batch 1024, plateau 5 / stop 15 on `det/f1_mean`. "v4b's recipe":
pick window always, real noise `std`, dropout 0.2, no rotation, N weight 1,
`log(softmax + eps)`, Adam 0.0001, batch 256, plateau 2 / stop 5 on
`val_loss`, seed 0; 12-20 min per run.)

- **Clean data: a tie.** On validation ours is within 0.002 of v4b (+0.001
  to +0.002). On test v4b leads P by 0.005 and trails S by up to 0.004;
  v4b's own recipe trained on our splits lands exactly where ours does
  (0.870 / 0.787), so v4b's P lead is what it gains from having trained
  on 69 % of the test traces.
- **Noisy data: ours wins.** +0.015 to +0.023 `noisy/det/f1_mean` over v4b
  on the same windows; v4b's recipe retrained here gains only +0.004.
- **Pick timing:** ours 0.153-0.157 s P MAE against v4b's 0.146 s; v4b's
  recipe here gives 0.154 s, so again the leakage, not the recipe.
- **Sigma 0.2 s beats 0.1 s** again (+0.008 val, +0.009 test S).
- **The partial-label mask in training** adds little: trained with
  `absent` and scored with `unread` (evaluation run of `f0c95f4e`), the
  model gets 0.867 / 0.794, against 0.871 / 0.790 and 0.869 / 0.786 with
  the mask: about +0.003 on P (higher precision), S within the seed noise.
  The mask matters most for the metrics, which no longer punish picking
  an unread P.
- Reproduction check: v4b's recipe in this framework scores v4b's own
  numbers (test F1 at the test-optimal thresholds 0.880 / 0.801 against
  v4b's 0.882 / 0.800).

## 2026-11-13: 60 s of context (50 Hz) on deeplowfrequency

MLflow experiment `deeplowfrequency/PhaseNet-60s`, code `e3e8026`. The
winning recipe of the v4b
comparison (rotate, real noise `peak`, dropout, `unread`) with 60 s input
windows, two ways: 50 Hz (volpick time-stretched, 3001 samples = 60 s;
`pick_window` 120 s) and 100 Hz with `augmentation.window_s` 60 (6000
samples, which PhaseNet accepts; 6001 it does not).

**Scoring across rates.** Window F1 depends on the window length, so all
models were scored two ways: `evaluate` on 60 s dev windows (100 Hz models
with `window_s` 60), and a whole-trace annotation script: every dev
trace (about 120 s per event trace) annotated whole with `model.annotate`,
as qseek runs the exported model, then scored with the same detection
metrics and the `unread` mask. The second is the deployment number.

Deployment (`annotate`, dev, F1 at the dev-optimal threshold):

| model | F1 P 0.5 / 0.2 s | F1 S 0.5 / 0.2 s | MAE P / S (found picks) |
| --- | --- | --- | --- |
| 50 Hz, sigma 0.4 s | **0.814 / 0.786** | 0.648 / 0.538 | **0.037** / 0.116 s |
| 50 Hz, sigma 0.2 s, seeds 42 / 43 | 0.797, 0.793 / 0.756, 0.751 | 0.644 / 0.549, 0.545 | 0.041 / 0.112 s |
| 30 s (v4 winners), seeds 42 / 43 | 0.777 / 0.736 | 0.646-0.651 / 0.570 | 0.050 / 0.090 s |
| 100 Hz trained on 60 s | 0.769 / 0.729 | 0.646-0.650 / 0.566 | 0.050 / 0.094 s |
| volpick | 0.766 / 0.726 | **0.667 / 0.581** | 0.050 / 0.092 s |
| v4b | 0.762 / 0.720 | 0.633 / 0.561 | 0.053 / 0.091 s |

- **50 Hz wins P in deployment**: +0.016 to +0.020 F1 at 0.5 s and +0.015
  to +0.020 at 0.2 s over every 100 Hz model (sigma 0.2 s, both seeds),
  +0.037 with sigma 0.4 s; P timing improves too. PhaseNet's receptive
  field is fixed in samples, so at 50 Hz it spans twice the time.
- **S does not gain**: at 0.2 s tolerance 50 Hz is 0.02 worse and its S
  timing 0.02 s worse. In deployment no fine-tuned model beats volpick on
  S: whole-trace annotation shows S precision of only 0.56-0.62, the
  fine-tuned models produce more false S peaks than volpick. Open.
- **100 Hz on 60 s windows** won on 60 s evaluation windows (dev F1 P 0.858
  against 0.831-0.835 for the 30 s models) but not in deployment: SeisBench
  fixes PhaseNet's `in_samples` at 3001 and leaves it out of the export, so
  `annotate` runs it on 30 s windows. Forcing `in_samples` 6000 breaks
  `annotate` (F1 0.05): it still maps the output to 3001 samples. 50 Hz is
  the deployable way to 60 s windows.
- The 60 s window scores (`evaluate`, dev): 50 Hz sigma 0.2 s 0.862 / 0.765,
  sigma 0.4 s 0.866 / 0.774; 30 s models and v4b 0.831-0.837 / 0.739-0.747.

## 2026-11-13: the new defaults on the new BohemiaSaxony

MLflow experiment `BohemiaSaxony-v2/PhaseNet-defaults`, code `e3e8026`.
The dataset was rebuilt:
329,173 traces (253k earthquake, 76k real noise of 120 s), 100 Hz ZNE,
102 GB; 44,883 traces have S only and 8,910 P only. From scratch, ZNE, Adam
0.003, otherwise the defaults (rotate, real noise `peak`, dropout, std,
`unread`, sigma 0.1 s, plateau 5 / stop 15); 50-67 epochs, about 1 h per
run with 256 GB. Its splits leak swarms: 75 % of the events have a
neighbour within 10 km and 60 s in another split, so the absolute scores
are optimistic; the differences stand. Validation `det/f1_mean` (the
`absent`-trained runs re-scored under `unread` with `evaluate` on the same
seed-42 windows):

| run (seeds 42 / 43) | val | noisy val | pick MAE P |
| --- | --- | --- | --- |
| defaults | **0.861 / 0.861** | **0.800 / 0.799** | 0.057 / 0.061 s |
| no real noise | 0.860 / 0.859 | 0.791 / 0.789 | 0.057 / 0.060 s |
| old defaults (no rotate, noise, dropout; peak; `absent`) | 0.857 | | 0.058 s |
| `absent` | 0.858 | | 0.056 s |
| cosine, 50 epochs, no early stop | 0.858 / 0.859 | 0.797 / 0.794 | 0.060 / 0.061 s |
| sigma 0.2 s | 0.863 | 0.803 | 0.065 s |
| `original`, no training | 0.701 | 0.603 | |

- **The new defaults hold** on a second dataset: +0.005 over the old ones.
- **Real noise** gains +0.009 to +0.010 on the noisy windows and is neutral
  on the clean ones (on deeplowfrequency: +0.06). A dataset that already
  has a quarter noise traces gains less.
- **The `unread` mask** gains +0.003 (deeplowfrequency: +0.003 on P).
- **Plateau with early stopping is stable here**: the two seeds differ by
  0.0004, and it beats a fixed 50-epoch cosine by 0.002-0.003. The
  instability seen on deeplowfrequency (one seed stopped after 24 epochs)
  belongs to the smaller dataset with its noisier validation, not to the
  augmentation. Keep plateau 5 / stop 15.
- **Label sigma depends on the events**: 0.2 s gains +0.0015 F1 here (within
  the noise) but costs 0.008 s P pick MAE; the impulsive tectonic onsets
  want 0.1 s (the default), the emergent DLF onsets 0.2 s.

## 2026-11-17: labelling every event in the window (`label_events`)

A metadata row holds only its own event. SeisBench's labeller, and ours
before `e6a62b0`, therefore label the arrivals of other events in a window
noise, and the metrics count their detections as false picks. Matched by
station and absolute time: 25 % of the BohemiaSaxony event traces have
another event's pick within 30 s of their own (deeplowfrequency 15 %; noise
traces none). Code `05557a3` / `ad01e9a` (branch `label-events`):
`augmentation.label_events` `trace_event` (the row) or `all_events` (every
event at the station with an arrival in the trace), the unread mask per event,
and `max_sp_s` to bound it. Arms, two seeds each: A `trace_event` (the old
runs), B `trace_event` + `max_sp_s` 15, C `all_events` + `max_sp_s` 15.
Everything scored with `evaluate` on the same seed-42 windows, under the old
ground truth (`trace_event`) and the full one (`all_events`).

**The old ground truth.** The old BohemiaSaxony model scored with all
arrivals: 21-23 % more true picks, a third of its false picks (2967 -> 1990
on dev P) are other events' arrivals, precision +0.06, F1 0.864 -> 0.902 (P),
and the F1-optimal threshold 0.475 -> 0.275. Validation scores and selected
thresholds of earlier runs are biased low and high. deeplowfrequency: F1 +0.01
(P) and +0.03 (S).

**BohemiaSaxony** (`BohemiaSaxony-v2/PhaseNet-label-events`; A is
`PhaseNet-defaults`; test `det/f1_mean`, mean of two seeds):

| arm | all events | trace event | pick MAE P / S (all events) |
| --- | --- | --- | --- |
| A `trace_event` | 0.888 | **0.853** | 19.0 / 29.8 ms |
| B + `max_sp_s` 15 | 0.889 | 0.853 | 19.4 / 30.8 ms |
| C `all_events` + `max_sp_s` 15 | **0.900** | 0.836 | **18.2 / 28.6 ms** |

**deeplowfrequency** (`deeplowfrequency/PhaseNet-label-events`, 50 Hz
volpick; A is `PhaseNet-60s`; test `det/f1_mean` under all events, seeds 42 /
43): sigma 0.3 s A 0.824 / 0.825, B 0.826 / 0.828, **C 0.831 / 0.831**; sigma
0.2 s A 0.823 / 0.824, B 0.823 / 0.825, **C 0.827 / 0.828**. S pick MAE about
3 ms better with C.

**qseek, deeplowfrequency** (Eifel network, 2021-11-01 to 2022-01-01,
sigma 0.3 s): C 23 / 24 deep located events and 19 / 21 in deep clusters, A
25 / 25 and 20 / 19, B 22 / 22 and 14 / 18; semblance median C 0.49-0.51,
A 0.48. About 25 events: a tie.

**qseek, West Bohemia swarm** (2025-12-04 to 2025-12-18, the reference search with only the picker
changed; located = uncertainty <= 2 km horizontal and 1 km vertical):

| picker | located | of the catalogue | semblance, 3058 common events | unc. h median |
| --- | --- | --- | --- | --- |
| `bohemia-saxony-retrained` (catalogue) | 3301 | 1.000 | 0.640 | 0.97 km |
| A seeds 42 / 43 | 4479 / 4215 | 0.974 / 0.972 | 0.820 / 0.807 | 1.10 km |
| B seeds 42 / 43 | 4102 / 4437 | 0.967 / 0.972 | 0.799 / 0.819 | 1.10 km |
| C seeds 42 / 43 | **5201 / 5119** | 0.955 / 0.955 | **0.926 / 0.894** | 1.14 km |

The reference picker's rerun reproduces the catalogue (5584 events). Every
model detects nearly all catalogue events: the "missed" ones (C 150, A 90)
are detected but located just above the cut-off. C's extra events have higher
semblance (median 0.58, A's 0.47) and magnitude than A's, and C does not split
events (4-6 pairs within 2 s).

- **Labelling every event wins**: +0.012 F1 on BohemiaSaxony, +0.004-0.006
  on deeplowfrequency, better pick times on both, and in the swarm 17 % more
  located events at 0.09 higher semblance. It costs precision only under the
  old ground truth, where its correct detections of other events count as
  false.
- **The mask bound alone does nothing** (B = A); it keeps the unread mask of
  one event from hiding the others' arrivals.
- **Default**: `label_events: all_events`, `max_sp_s: 15` (local and volcanic
  data; raise it for regional data). Compare earlier runs only under one
  ground truth: re-score them with `evaluate`, which now uses all events.

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
