# Findings

What our experiments established, and the defaults they set. Each entry
links to the full record in the [experiment log](experiments.md), with the
MLflow experiment and the commit. Datasets: BohemiaSaxony (329k traces of
tectonic swarm earthquakes in West Bohemia and Saxony) and deeplowfrequency
(130k traces of deep low-frequency events from volcanic regions).

## How to compare runs

- **Run-to-run noise** with the same config and seed is about 0.002
  `det/f1_mean` and 4 ms pick MAE (GPU non-determinism). Compare settings
  with two seeds; one seed only resolves effects above about 0.005.
- **Select and compare by `det/f1_mean`**, not `val_loss`: the loss is not
  comparable between runs whose loss settings differ.
- **Compare under one ground truth.** What counts as a true pick depends on
  `augmentation.label_events` and `missing_picks`; re-score old models with
  `evaluate` before comparing them with new ones.
- **Mind the splits.** On BohemiaSaxony, 25 % of the dev and test traces
  share seconds of waveform with training traces (swarm events overlap in
  time), so its absolute scores are optimistic; differences between runs
  stand.

## The defaults and why

| Setting | Default | Evidence |
| --- | --- | --- |
| Labels | `label_events: all_events`, `max_sp_s: 15` | +0.012 test F1 on BohemiaSaxony, +0.004-0.006 on deeplowfrequency, better pick times; 17 % more located events in a qseek swarm benchmark ([2026-11-17](experiments.md#2026-11-17-labelling-every-event-in-the-window-label_events)) |
| Unread phases | `missing_picks: unread` | +0.003 on both datasets; the metrics no longer punish picking an unread P |
| Loss | cross-entropy from logits, noise weight 0.5 | best F1 on BohemiaSaxony (+0.004 over weight 1), higher F1-optimal thresholds (about 0.5 against 0.3) ([2026-11-12](experiments.md#2026-11-12-phasenet-loss-on-bohemiasaxony)) |
| Run length | plateau 5, early stop 15, `min_delta` 0.002, at most 100 epochs | the best of the run-length variants ([2026-11-12](experiments.md#2026-11-12-run-length-without-a-fixed-epoch-count)); two seeds within 0.0004, 0.002-0.003 above a fixed cosine schedule ([2026-11-13](experiments.md#2026-11-13-the-new-defaults-on-the-new-bohemiasaxony)) |
| Selection | `training.monitor: det/f1_mean` | comparable across loss settings |
| Learning rate | Adam 0.001 | ties 0.0005 when fine-tuning, 0.01 is worse |
| Augmentation | pick window 2/3, rotation 0.5, real noise 0.5 (`peak`), channel dropout 0.2 | +0.005 over the old recipe on BohemiaSaxony; real noise gains 0.06 F1 on noisy validation windows for 0.005 on clean ones |
| Label width | `label_sigma_s: 0.1` | right for impulsive tectonic onsets; emergent onsets want more (below) |

## Per dataset

- **Tectonic swarms (BohemiaSaxony)**: the defaults. Training from scratch
  and fine-tuning `original` end within the noise (0.846 and 0.847). The
  pretrained `original` PhaseNet scores 0.70 without training.
- **Deep low-frequency events (deeplowfrequency)**: fine-tune `volpick`
  (+0.017 over training from scratch, +0.023 over volpick itself). Use a
  wider label, 0.2-0.4 s: sigma 0.2 s is the best F1 / pick-time trade-off
  offline. For catalogues, run the model at 50 Hz (60 s windows, volpick
  time-stretched) with sigma 0.3-0.4 s: in qseek it locates 25-27 deep
  events against 15-17 with volpick or the v4b reference picker.

## What did not work

- **Masking the loss to the surroundings of the picks** lowered F1 at every
  width (-0.03 to -0.09): the model never learns that P and S coda are not
  picks. Removed.
- **A fixed long schedule**: 100 epochs at a constant rate ended below a
  40-epoch cosine run; the decaying rate matters more than the length.
- **Bounding the unread mask alone** (`max_sp_s` without `all_events`)
  changes nothing.

## Open

- Confirm the noise weight 0.5 with several seeds and when fine-tuning.
- Check in snuffler the events only the `all_events` model locates in the
  qseek swarm benchmark (most look real: they lie in the swarm volume and
  follow its rate).
- Mixed precision against `32-true`.
