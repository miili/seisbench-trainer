# Requirements: seisbench-trainer

Status: draft for review (2026-11-08). The IDs (`FR-*`, `NFR-*`, `D-*`) are
the references for the implementation plan.

## 1. Goal

A small, maintainable workbench to train, fine-tune and evaluate SeisBench
phase pickers with PyTorch Lightning. It logs everything to MLflow 3.x. It
runs unchanged on a laptop (CPU, small data) and on a Slurm GPU cluster (one
GPU per job, large data). Hyperparameter optimisation (HPO) with Optuna is
optional.

It replaces the training part of `../khuzaima` (`seisbench_training`).

## 2. Environments and constraints

| Item | Laptop (development) | Slurm cluster (production) |
| --- | --- | --- |
| Hardware | small machine, CPU only | one GPU per job, large RAM |
| Python environment | `uv` | `conda` env; the package is installed with `pip install -e .` |
| Data | small SeisBench dataset or synthetic data | full datasets, for example BohemiaSaxony (about 160 GB in RAM) |
| MLflow | `sqlite:///mlflow.db` (MLflow default) | **folder store on the shared filesystem**, `file:///<shared>/mlruns` with `MLFLOW_ALLOW_FILE_STORE=true`, no MLflow server (see D-1) |
| Optuna storage | in-memory or SQLite | journal file on the shared filesystem (see FR-H3) |

Fixed decisions:

- Configuration: **Pydantic v2**. Hydra is not used.
- Python: **3.13**, on the laptop (`uv`) and on the cluster (conda).
- Datasets are **already downsampled** to the target sampling rate. No
  resampling or conversion step is needed.
- Training loop: **PyTorch Lightning**.
- HPO: **Optuna**, installed only with an optional extra
  (`pip install seisbench-trainer[hpo]`).
- Data: **always preloaded** into RAM. Loading from file is the bottleneck,
  not the GPU.
- Compute: **one GPU per run**. No DDP or multi-node training.
- MLflow: **`mlflow>=3`**.

## 3. Functional requirements

### 3.1 Configuration

- **FR-C1:** One Pydantic model (`TrainConfig`) describes a run. Its
  sections are `data`, `model`, `augmentation`, `training`, `evaluation`,
  `tracking` and the optional `hpo`.
- **FR-C2:** A config is one JSON file per run, loaded with
  `TrainConfig.model_validate_json()`. There is no YAML, no layering and no
  CLI overrides. `seisbench-trainer config` writes a complete default config
  as a starting point.
- **FR-C3:** Check the config when it loads. Wrong types, unknown keys
  (`extra="forbid"`), `num_workers < 1`, invalid component orders, and window
  lengths that do not fit the model are errors.
- **FR-C4:** Write a JSON schema of the config, for editor completion and
  documentation.
- **FR-C5:** Log the fully resolved config to every MLflow run as an
  artifact (`config.json`) and as flattened params. A run must be reproducible
  from that artifact alone (`seisbench-trainer train --config config.json`).
- **FR-C6:** Lengths and tolerances are given in seconds in the config. The
  code converts them to samples in one place, using the dataset sampling
  rate.
- **FR-C7 Discriminated unions:** wherever the config chooses between
  alternatives, use a Pydantic discriminated union. Each alternative is its
  own model with a `Literal` tag field, and the union uses
  `Field(discriminator=…)`. Examples:
  - models: `PhaseNetConfig | EQTransformerConfig`, tagged by `type`;
  - datasets: `SeisBenchDataset | LocalDataset`, tagged by `type`;
  - optimizers: Adam / AdamW;
  - LR schedulers: none / cosine / plateau;
  - HPO samplers, pruners and search-space parameters (float / int /
    categorical).

  Each alternative builds its own runtime object, for example
  `PhaseNetConfig.build() -> sbm.PhaseNet`. To add an alternative, write one
  new class and add it to the union. There is no separate registry dictionary
  and no `if name == …` dispatch.

### 3.2 Data

- **FR-D1:** `data.datasets` is a list of one or more datasets. Each entry
  is a `seisbench.data` class name, or a local SeisBench dataset path, with
  optional metadata filters. The component order and the sampling rate are
  set once for the whole run, and every dataset is loaded with them.
- **FR-D2:** Split each dataset with its own `train_dev_test()`, then
  concatenate the splits per type (train with train, dev with dev, test with
  test). Optional metadata filters (for example by network or magnitude)
  apply per dataset, before the split.
- **FR-D2a Per-dataset metrics:** when a run combines several datasets, the
  validation and test metrics are logged for the combined set and for each
  dataset separately, with the dataset name in the metric key. One dataset
  must not hide poor performance on another.
- **FR-D2b Weighting (optional):** a per-dataset sampling weight for
  training, so that a small dataset is not swamped by a large one. The
  default is no weighting (plain concatenation).
- **FR-D3:** Always preload the waveforms into RAM, once per process. The same
  loaded data must serve several training runs in one process (HPO trials).
- **FR-D4:** DataLoader workers must share the preloaded arrays and not copy
  them. Use fork, copy-on-write, and persistent workers.
- **FR-D5:** Provide a synthetic dataset source (`{"type": "synthetic"}`)
  in the package, for tests and laptop runs. It downloads nothing.
- **FR-D6:** Log the dataset identity as an MLflow dataset input
  (`mlflow.log_input`). Record the name, component order, sampling rate,
  number of traces per split and a digest of the metadata.

### 3.3 Models (agnostic, PhaseNet first)

- **FR-M1:** Support `sbm.PhaseNet` (first milestone) and
  `sbm.EQTransformer` (second milestone). Adding another SeisBench picker
  with sample-wise outputs (for example PhaseNetLight or OBSTransformer) must
  only require one small model adapter and no changes elsewhere.
- **FR-M2:** The model adapter *is* the model's config class, a member of
  the model discriminated union (FR-C7). It lives in its own module under
  `models/` and subclasses an abstract base. It defines only what differs
  between models:
  - how to build the model (from scratch, or from pretrained weights given by
    name or local path);
  - the label augmentations (for example EQTransformer also needs a
    detection label);
  - the loss;
  - the mapping from model output to per-phase probability curves (P, S).
- **FR-M3:** The input window length comes from the model (`in_samples`)
  unless the config overrides it.
- **FR-M4:** Read the label and component order from the model
  (`model.labels`, `model.component_order`). Never hard-code channel indices.
- **FR-M5:** The saved model metadata must give the component order that the
  model was trained on. Check this before export.

### 3.4 Training

- **FR-T1:** Use a LightningModule for the training and validation steps.
  The optimizer and an optional LR scheduler come from the config.
- **FR-T2:** Lightning callbacks: `ModelCheckpoint` on the selection metric
  (one mechanism only), optional `EarlyStopping`, and LR monitoring.
- **FR-T3:** The accelerator, devices and precision come from the config.
  The default is `auto`, so the same config runs on CPU and GPU.
- **FR-T4:** Set the seed for everything (`seed_everything(workers=True)`)
  and log it.
- **FR-T5:** The selection metric (for example `val_loss`, or the mean of
  P and S validation F1) can be set in the config. Checkpointing, early
  stopping, HPO and the model registry all use the same metric.

### 3.5 Evaluation

Two metric families answer two questions. Both are computed per phase on the
validation set after every epoch and logged with the epoch as the step.

- **FR-E1 Pick accuracy (window-based; `calculate_pick_differences` in the
  reference code, metric prefix `pick_`):** how accurately does the model
  time a pick that is known to be present? For each labelled pick, only the
  prediction inside a window of ±`pick_window_s` around the *expected*
  (labelled) pick is considered. Everything outside it is discarded. The
  predicted pick is the highest point of the prediction in the window. The
  window is given in seconds in the config. Outputs:
  - residual statistics: mean, median, MAE, RMS, the 5/25/75/95
    percentiles, the median absolute deviation (MAD) and the fraction of
    residuals within each tolerance;
  - ROC AUC, and precision/recall/F1 against the threshold, using the peak
    value in the window as the score for a pick and the peak value on
    pick-free traces as the score for noise.

  The residual statistics are given both without a threshold and at the
  selected threshold.
- **FR-E2 Detection (label-free; `extract_picks` in the reference code,
  metric prefix `det_`):** how well does the model find picks with no prior
  knowledge? Detect peaks (`find_peaks`) over the whole window, match them
  one-to-one with the manual picks within a tolerance, count unmatched peaks
  as false picks and unmatched labels as misses. Outputs: precision, recall
  and F1 against the threshold, the F1-optimal threshold, and the residuals
  of matched picks.
- **FR-E2a Fixes and improvements to the reference code:**
  1. The edge mask (`p_mask_noise[:edge_mask] = False`) indexes *traces*,
     not samples. It drops the first 100 traces of each batch from the noise
     set. Replace it with a sample-based mask at the trace edges, given in
     seconds.
  2. A pick is "present" when the label peak is ≥ 0.5, not when it is
     > 0. Otherwise the Gaussian tail of a pick just outside the window
     counts as a pick at the window edge.
  3. Take the true pick sample from the metadata, which the SeisBench
     augmentations keep up to date. It is passed through the generator as an
     extra key. This replaces the `argmax` of the label curve, so every pick
     in the window is found, not only the largest.
  4. Several picks of the same phase in one window (for example
     aftershocks) are each evaluated with their own window. Where windows
     overlap, each sample is assigned to the nearest labelled pick.
  5. `scalar_metrics()` reads `self.sampling_rate` but is a module-level
     function (`NameError`). Pass the sampling rate as an argument.
  6. Precision, recall and F1 handle zero denominators without NaN
     warnings. The F1-optimal threshold ignores NaN values.
  7. Index channels by phase name from `model.labels`, not with a fixed
     `ORDER_MAP` of `"PSN"` and `"NPS"`. This is needed for EQTransformer and
     other models.
  8. The evaluation uses vectorised NumPy over the whole epoch instead of a
     Python loop per trace where possible (`extract_picks` loops over
     traces).
- **FR-E2b Unbiased validation windows:** the reference code centres every
  validation window on the first pick (`WindowAroundSample`,
  `selection="first"`). P therefore always sits at the window centre. The
  validation windows must instead use a random offset with a fixed seed, so
  that they are deterministic across epochs and runs but not biased to one
  position.
- **FR-E2c Noise examples:** the validation windows are cut around picks, so
  there are very few pick-free traces, and ROC and precision rest on few
  negatives. Add explicit noise examples: the dataset's noise traces if it
  has them, otherwise a window that ends before the P arrival. Report how many
  negatives were used.
- **FR-E3:** Optional baseline: run a pretrained SeisBench model (default
  `original`) alongside, with its inputs reordered to its own component
  order. Log it under its own metric prefix and include it in the comparison
  plots.
- **FR-E4:** Figures (residual histogram, PR/F1 against threshold, ROC,
  baseline comparison) are logged as step-indexed images
  (`mlflow.log_image(key=…, step=…)`), not as one file per epoch. Create them
  every *n* epochs (config option; 0 means off).
- **FR-E5:** Test evaluation runs only on request: at the end of training,
  or as a separate command. Use several tolerances, and the thresholds
  selected on the validation set. Log the results as metrics and as a table
  (`mlflow.log_table`).
- **FR-E5a Cross-dataset evaluation:** `evaluate` accepts any model and any
  dataset and split. The model can be registered (`models:/…`), a SeisBench
  pretrained name, or a local path. Link the results to the model and the
  dataset (MLflow 3: `log_metric(..., model_id=…, dataset=…)`), so that
  `search_logged_models` can rank models per dataset.
- **FR-E6:** The evaluation code is plain NumPy/PyTorch functions, separate
  from the Lightning callback, so it is reusable offline and unit-testable.

### 3.6 Tracking and model management (MLflow 3)

- **FR-L1:** The code only uses `MLFLOW_TRACKING_URI` (or `tracking.uri`).
  The code is the same for SQLite, the file store and a server.
- **FR-L2:** Each run records the resolved config, git commit and dirty
  flag, package versions (torch, seisbench, lightning, mlflow), host, Slurm
  job and array IDs, and tags (`purpose`, `model`, `dataset`, `study`).
- **FR-L3:** Enable system metrics (CPU, GPU, RAM) once per run.
- **FR-L4:** Log the best model as an MLflow **LoggedModel**, with its step
  and metrics linked through `model_id`. Attach the SeisBench `.json`/`.pt`
  export, so it can be loaded with `sbm.<Model>.load()` and used directly in
  Qseek.
  Note: since MLflow 3.18, `mlflow.pytorch.log_model` uses the `pt2` format
  by default and then requires an `input_example`.
- **FR-L5:** Model registry: one command registers a run's best model under a
  name and sets an alias (for example `@champion`). One command downloads a
  SeisBench model by `models:/<name>@<alias>`.
- **FR-L6:** Do not log Lightning checkpoints to MLflow by default. They stay
  in a local or scratch directory.
- **FR-L7:** Use async logging (`enable_async_logging`) so that logging does
  not block training.
- **FR-L8 Naming conventions** (generated, can be overridden):
  - one experiment per dataset and task, for example
    `BohemiaSaxony/phasenet`;
  - run names `<variant>-<short hash of the config>`;
  - registered model names `<model>-<dataset>[-from-<pretrained>]`, the
    same scheme as the SeisBench export name.

  Many models from many datasets must stay easy to find and compare.

### 3.7 Hyperparameter optimisation (optional extra)

- **FR-H1:** HPO is only available with the `[hpo]` extra. Without the extra,
  everything else works, and the HPO command explains how to install it.
- **FR-H2:** The search space is defined in the config (parameter, type,
  range, log scale, choices). It is not written in code.
- **FR-H3:** Distributed execution without a server: N Slurm array tasks,
  each one a worker running `study.optimize()` against a shared Optuna
  `JournalStorage` file (with a lock that works on NFS) on the shared
  filesystem.
- **FR-H4:** Each worker preloads the data once and runs trials one after
  another in the same process (FR-D3).
- **FR-H5:** Pruning of bad trials (MedianPruner or ASHA) uses the per-epoch
  selection metric. Pruning can be switched off.
- **FR-H6:** MLflow structure: one parent run per study and one child run
  per trial, tagged with the study name and trial number. The parent run holds
  the best parameters and Optuna summary plots.
- **FR-H7:** Workers can be stopped and restarted. Failed or pruned trials do
  not stop the study.

### 3.8 Command-line interface

- **FR-CLI1:** The CLI is built with **Typer**, with typed `Annotated`
  options. Subcommands:
  - `train`;
  - `evaluate` (test set, given a run or a registered model);
  - `hpo` (only with the extra);
  - `register`;
  - `export` (writes the SeisBench model to a directory);
  - `config` (prints the default config or its schema).
- **FR-CLI1a Console output with Rich:** logging goes through
  `rich.logging.RichHandler`. Result summaries (metrics, HPO results) are
  printed as Rich tables. Training uses Lightning's `RichProgressBar`. Plain
  output for Slurm logs is the default when no terminal is attached, which is
  what Rich does on its own.
- **FR-CLI2:** Example Slurm scripts (`sbatch`), one for a single training
  run and one for an HPO worker array, using the conda environment.
- **FR-CLI3 Batches of runs:** a Slurm array script that takes a text file
  listing config files (one per line). Training *N* models on *N* datasets
  is then one `sbatch` call, with no extra tooling.

## 4. Non-functional requirements

- **NFR-1 Simplicity and modularity:** one subpackage per concern:
  `data`, `models`, `metrics`, `tracking`, `hpo`. The dependencies point one
  way:
  - `metrics` depends on nothing in the package;
  - `models` and `data` depend only on the config;
  - `tracking` and `train` put the pieces together.

  New models, datasets and optimizers are added as new union members
  (FR-C7), not by editing existing code. There is no plugin framework and no
  entry-point discovery. A model module is about 50 to 100 lines. Use
  Lightning, MLflow and Optuna features rather than custom re-implementations.
- **NFR-2 Reproducibility:** each run can be repeated from its MLflow
  record (FR-C5, FR-L2).
- **NFR-3 Laptop-friendly:** the full test suite and a smoke training run
  (CPU, about 1 minute, synthetic or tiny data) run on a small laptop.
- **NFR-4 Throughput:** after preloading, GPU use must not be limited by
  augmentation in the DataLoader. Measure and log the data and step time.
- **NFR-5 Robustness on Slurm:** a job that times out or is cancelled leaves
  its MLflow run marked `KILLED` or `FAILED`, not `RUNNING`. HPO workers can
  be restarted.
- **NFR-6 Dependencies:** the core needs torch, lightning, seisbench, mlflow,
  pydantic, typer, rich, numpy, scipy, scikit-learn and matplotlib.
  Optuna and optuna-integration are only in `[hpo]`. Ruff and pytest are dev
  dependencies.
- **NFR-7 Quality:** ruff lint and format pass. There are unit tests for the
  evaluation metrics (with hand-built label and prediction arrays and
  known answers), the config validation and the model adapters, plus an
  end-to-end smoke test (FR-D5).
- **NFR-8 Typing:** the codebase is fully typed: every function signature
  and every class attribute. Use **pyright** in `strict` mode on `src/` and
  `tests/` (configured in `[tool.pyright]` in `pyproject.toml`), and ship a
  `py.typed` marker. Untyped third-party packages (seisbench, possibly
  others) are handled with narrow settings for those packages only, not by
  relaxing strict mode globally. Avoid `Any` unless a third-party boundary
  needs it. Use `numpy.typing.NDArray` and `torch.Tensor` for arrays.
- **NFR-9 Pre-commit:** `.pre-commit-config.yaml` with `ruff check --fix`,
  `ruff format`, pyright (a local hook run in the project environment, so it
  sees the dependencies), and the standard hooks (`trailing-whitespace`,
  `end-of-file-fixer`, `check-yaml`, `check-toml`, `check-added-large-files`).
  Ruff rules: at least `E`, `F`, `I`, `UP`, `B`, `SIM`, `ANN`, `RUF`. The rule
  set is configured in `pyproject.toml`.
- **NFR-10 Generic framework:** training a new PhaseNet model on a new
  dataset must need only a new JSON config: copy an existing one and change
  the dataset entry. No code changes.
  Training, HPO and evaluation must use the same config and the same code
  path.

## 5. Out of scope

- Running Qseek, and catalogue analysis.
- Multi-GPU or multi-node training (DDP).
- Streaming data from disk (data is always preloaded).
- Model serving, and MLflow GenAI or tracing features.
- Thesis notebooks and figures.
- Hydra compatibility or migrating the old `mlruns`.

## 6. Decisions

### D-1: MLflow storage — folder store on the shared filesystem (decided)

The cluster uses `file:///<shared>/mlruns` with
`MLFLOW_ALLOW_FILE_STORE=true`. Each run writes to its own folder, which is
safe when many Slurm jobs write at the same time. Tested with MLflow 3.18:
LoggedModel, metrics linked to a model, and the registry with aliases all
work on the folder store.

Rejected options:

- SQLite on the shared filesystem: its locking is unreliable on network
  filesystems.
- A per-job SQLite file merged afterwards: it needs a merge step and gives
  no combined view.
- An `mlflow server` running as a Slurm job, optionally backed by Postgres:
  this needs infrastructure.

Migration path: the folder store is in maintenance mode. When MLflow drops
the `MLFLOW_ALLOW_FILE_STORE` opt-out, or the UI becomes too slow, move with
`mlflow migrate-filestore` to SQLite behind an `mlflow server`, or to
Postgres. The code only reads `MLFLOW_TRACKING_URI` (FR-L1), so no code
changes are needed. To view results, run `mlflow ui` on a login node behind
an SSH tunnel, or copy the store to the laptop with rsync.

### D-2: Python 3.13 (decided)

Python 3.13 everywhere, installed with `uv` on the laptop and conda on the
cluster.

### D-3: One GPU per run (decided)

There is no DDP. HPO scales out with more Slurm array tasks (FR-H3), not with
more GPUs per run.

### D-4: Combined datasets (decided)

One run can train on several datasets at once (FR-D1, FR-D2, FR-D2a, FR-D2b).

### Still to check on the cluster

- The type of the shared filesystem (`df -T <path>`: NFS, Lustre or GPFS).
  The Optuna journal lock depends on it (FR-H3).

Known numbers: preloading takes a few minutes and a training run about an
hour. The cluster has plenty of storage, so the artifact volume is not a
constraint.
