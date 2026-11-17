---
icon: lucide/database
---

# Data

## Sources

`data.datasets` lists one or more datasets; their train, dev and test
splits are concatenated.

| `type` | Source |
| --- | --- |
| `seisbench` | A class of `seisbench.data` by name, e.g. `ETHZ`, downloaded on first use |
| `local` | A SeisBench-format folder (`metadata.csv`, `waveforms.hdf5`) |
| `synthetic` | Generated noise with P and S wavelets, for tests and smoke runs |

Each source can select traces with a pandas `query` on its metadata, e.g.
`"source_magnitude > 1"`, and apply `preprocessing` steps before the split:

- `filter`: keep the traces that match a query, in some splits only. For
  example `trace_category == "noise"` on `["train"]` trains on noise traces
  only while validation and test stay whole.
- `resplit`: replace the dataset's split with a seeded random one. Traces of
  one event (`group_by`, default `source_id`) stay in one split, so an event
  recorded at many stations is not in both training and test.

Never let a hyperparameter search change preprocessing: that would change the
test set.

## Bring your own dataset

A local dataset is a folder in SeisBench's format: `metadata.csv` with one
row per trace and `waveforms.hdf5` with the waveforms. SeisBench's
`WaveformDataWriter` writes both
([SeisBench docs](https://seisbench.readthedocs.io)). The trainer reads
these metadata columns:

| Column | Needed for |
| --- | --- |
| `trace_name`, `trace_sampling_rate_hz` | Every dataset |
| `split` (`train`, `dev`, `test`) | The splits; or add a `resplit` step |
| `trace_<phase>_arrival_sample`, e.g. `trace_P_arrival_sample`, `trace_Sg_arrival_sample` | The picks; recognised phase names: `P`, `p`, `Pg`, `Pn`, `P1`, `pP`, `PmP`, `pwP`, `pwPm` and `S`, `s`, `Sg`, `Sn`, `S1`, `SmS` |
| `trace_start_time`, `station_network_code`, `station_code` (and `station_location_code`, `source_id` if present) | The default `label_events: all_events`; without them set `"label_events": "trace_event"` |
| `trace_category` with noise traces as `"noise"` | Real-noise augmentation and the noisy validation; skipped with a warning without noise traces |

`seisbench-trainer config` checks the config, not the data: a missing column
shows up when a run loads the dataset.

## Component order and sampling rate

`data.component_order` is the channel order the model is trained on, and the
exported model states it. Pretrained weights need their own order (`original`
is `ENZ`, `volpick` is `ZNE`); a mismatch is an error, because feeding another
order swaps the channels the weights expect.

`data.sampling_rate` is the rate every dataset is resampled to on access;
pick positions follow. Decimation is zero-phase by default
(`resample_zerophase`), as in SeisBench's `annotate`, so training and
inference see the same filter.

Pretrained weights run at their own rate. To run them at another one, set
`model.sampling_rate` to the data rate: the weights then see data at the new
rate as if it were at the old one, which stretches time (a 100 Hz PhaseNet at
50 Hz sees 3001 samples = 60 s). The exported model stores the new rate, so
`annotate` and qseek resample their input to it.

## Memory

All splits are preloaded into memory before training. Plan the job memory for
the decoded waveforms at `data.sampling_rate`: BohemiaSaxony (329k traces of
120 s, 3 components at 100 Hz) needs about 160-256 GB.

## Training windows

`augmentation.train` is the training pipeline, applied in order:

1. `pick_window` steps (optional): cut a context window around a random pick,
   with `probability`; otherwise the next step crops from the whole trace.
2. Exactly one `random_window`: crop a random window of the model input
   length.
3. Perturbations, each with a `probability`: `rotate_horizontal`,
   `real_noise`, `channel_dropout`, `add_gap`, `gaussian_noise`.

The defaults follow SeisBench's pick-benchmark PhaseNet recipe plus the
perturbations that held up in our experiments ([Findings](../findings/index.md)):

```json
[
  {"type": "pick_window", "probability": 0.667},
  {"type": "random_window"},
  {"type": "rotate_horizontal", "probability": 0.5},
  {"type": "real_noise", "probability": 0.5},
  {"type": "channel_dropout", "probability": 0.2}
]
```

The perturbations are reviewed re-implementations of the
`seisbench.generate` ones; each class documents where it deviates (for
example, `real_noise` uses only the training split's noise traces, rotates
their horizontals and never adds noise to a missing component). Datasets
store missing components and gaps as NaN; every pipeline sets them to zero
first, as `annotate` does.

Validation and test use fixed windows instead: the first pick at a random but
per-trace fixed offset, so the windows are identical in every epoch and run
with the same seed.
