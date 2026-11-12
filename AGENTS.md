# AGENTS.md

Guidance for coding agents working in this repository.

## Purpose

`seisbench-trainer` is a training workbench for seismological phase pickers
built on [SeisBench](https://github.com/seisbench/seisbench), PyTorch
(Lightning) and [MLflow](https://mlflow.org) 3.x. It trains or fine-tunes
pickers (PhaseNet first) on any `seisbench.data` dataset, evaluates them
after every epoch, logs everything to MLflow, and runs hyperparameter
optimisation (HPO) on a workstation or a Slurm cluster.

## Status

This is a new project. Only the `uv init` scaffold exists. The training code
is being ported from the MSc thesis project in `../khuzaima`
(package `seisbench_training`). Requirements and the plan are defined before
any code is written. Do not add modules ahead of the agreed plan.

The requirements are in `docs/requirements.md`. The step-by-step
implementation plan is in `docs/plan.md`; implement one step at a time.
Training experiments that set defaults (loss, weights, selection metric)
are recorded in `docs/experiments.md`; read it before changing a default.
Decisions already made:

- Pydantic v2 for configuration, one JSON file per run. No Hydra, no YAML,
  no config layering.
- Every choice between alternatives (model, dataset source, optimizer,
  scheduler, HPO sampler, pruner, search-space parameter) is a Pydantic
  discriminated union. Each member is a class with a `Literal` tag and
  builds its own runtime object (`.build()`, `.load()`). To extend the code,
  add a member to the union. Do not add `if`/`elif` dispatch or a registry
  dictionary.
- Modular layout: one subpackage per concern (`data`, `models`,
  `training`, `metrics`, `tracking`, `hpo`). `metrics` has no internal
  dependencies, and `cli` sits on top. The architecture is in
  `docs/plan.md`.
- CLI with Typer. All console output through Rich: `RichHandler` logging,
  Rich tables, Lightning `RichProgressBar`. Only `cli.py` imports `typer`;
  only `console.py` configures Rich.
- PyTorch Lightning for training.
- Optuna for HPO, but only as the optional `[hpo]` extra.
- Data is always preloaded into RAM.
- One GPU per run.
- PhaseNet first. The code must stay model-agnostic: EQTransformer and
  other pickers are added through small model adapters.
- Development on a small laptop (CPU only). Production on a Slurm GPU
  cluster with conda and a shared filesystem, and no MLflow server.
- Python 3.13 everywhere.
- One run can train on several datasets at once. Datasets are already
  downsampled.

## Reference project: `../khuzaima`

Read this code. Do not import from it or copy it unchanged.

| File | Contents |
| --- | --- |
| `src/seisbench_training/main.py` | Hydra entry point: dataset loading, augmentations, DataLoaders, `MLFlowLogger`, `pl.Trainer` |
| `src/seisbench_training/utils/model_utils.py` | `SeisBenchLit` LightningModule (PhaseNet only), `phase_dict`, cross-entropy loss |
| `src/seisbench_training/metrics/evaluation_metrics.py` | `PickStats`, peak extraction (`extract_picks`), precision/recall/F1 vs threshold, ROC/AUC, matplotlib plots |
| `src/seisbench_training/metrics/callbacks.py` | `EvaluationMetrics` callback: per-epoch metrics and figures, baseline comparison, best-model export, test evaluation at the end |
| `src/seisbench_training/configs/` | Hydra config groups: `dataset`, `model`, `training`, `augmentations`, `evaluation` |
| `slurm/*.sbatch` | HPO grid (Hydra `-m` multirun inside a Slurm array) and final training |
| `scripts/evaluate_final_models.py` | Offline test-set evaluation of exported models |

Known defects and limits in the reference code. Fix these when porting:

- `scalar_metrics()` is a module-level function, but it reads
  `self.sampling_rate`. This raises a `NameError`.
- In `calculate_pick_differences`, the edge mask indexes traces instead of
  samples, so the first 100 traces of each batch are dropped from the noise
  set. "Pick present" is tested with `label > 0` instead of `>= 0.5`. Only
  the largest pick per phase and trace is evaluated. Requirement FR-E2a in
  `docs/requirements.md` lists all the fixes.
- Validation windows are always centred on the first pick, so P sits at the
  window centre.
- The best model is stored twice: once by `ModelCheckpoint` and once as an
  in-memory `deepcopy` in the callback. `MLFlowLogger(log_model=True)` also
  uploads the Lightning checkpoints.
- System metrics are enabled twice: `enable_system_metrics_logging()` and a
  manual `SystemMetricsMonitor`.
- `accelerator="gpu"` and `devices=1` are hard-coded.
- Only PhaseNet is supported. The baseline model is always `sbm.PhaseNet`.
- The config keys `patience`, `optimizer`, `model.name` and
  `dimension_orders` exist but are not used.
- The test loader shuffles.
- `scikit-learn` is imported but is not a declared dependency. `scipy` is
  declared only in an optional extra.
- Every epoch logs about 12 PNG figures, which adds up to a large number of
  artifacts.
- The full dataset is preloaded into RAM (160 GB for BohemiaSaxony). An
  `lru_cache` reuses it across in-process Hydra multirun jobs.

## Domain rules (do not break)

- **Component order:** the waveform component order (`ZNE`, `ENZ`, …) that
  the model is trained on must match the order in the saved model's `.json`
  metadata. It must also match the order used at inference. Pretrained
  SeisBench PhaseNet uses `ENZ`. When you evaluate a baseline, reorder the
  input channels to the baseline's `component_order`.
  Fine-tuning must use the pretrained model's own order. A mismatch is an
  error. The reference project fine-tuned the `ENZ` model on `ZNE` data,
  which swaps the input channels; do not reproduce that.
- **Label order:** the model's `labels` (for example `PSN`) decide which
  output channel is P, S or noise. Always map channels by name, never by
  a fixed index.
- **Sampling rate:** the dataset sampling rate, the window lengths in samples,
  the tolerances in seconds and the peak distances must stay consistent.
  Convert seconds to samples once, in one place, always with
  `data.sampling_rate`. A model runs at the data rate: pretrained weights at
  another rate need an explicit `model.sampling_rate` (time-stretching, see
  FR-M5b), never a silent mismatch. Decimate zero-phase
  (`data.resample_zerophase`), as `annotate` does.
- **Splits:** use the dataset's own `train_dev_test()` split, unless the
  dataset's `preprocessing` has a `resplit` step. A resplit keeps events
  together (`group_by`) and is seeded, so the test set is fixed for a config.
  Do not let HPO vary preprocessing: that changes the test set. Never use the
  test split for model selection, threshold selection or HPO.
- **Thresholds:** select detection thresholds on the validation set and apply
  them unchanged to the test set.
- **Exported models:** save the best model as a SeisBench `.json`/`.pt` pair
  (`model.save(path)`), so that `sbm.<Model>.load()` and Qseek can use it
  directly.

## Tooling

- Python 3.13 (`.python-version`). Package manager: `uv`. Build backend: setuptools with `setuptools-scm`;
  the version comes from git (no version string in the code).
- Source layout: `src/seisbench_trainer/`.
- Install the environment: `uv sync`
- Run the entry point: `uv run seisbench-trainer ...`
- Laptop smoke run: `uv run seisbench-trainer train configs/synthetic-smoke.json`
  (synthetic data, CPU, under a minute; writes to `./mlflow.db`).
- Every file in `configs/` is validated by the tests; keep them valid when
  the config schema changes.
- Lint and format:
  `uv run pre-commit run --all-files` (ruff check, ruff format).
- Full inline type annotations are required in our code: every function
  signature and class attribute (ruff `ANN` enforces the signatures). There
  are no separate stub files; do not add `.pyi` stubs for third-party
  packages such as seisbench or `optuna_integration`.
- Import seisbench (it pulls in torch) only where data or models are
  actually loaded, so that config handling and the CLI stay fast.
- Tests use the in-package synthetic dataset; they must not download data.
  Tests that need the pretrained `original` PhaseNet skip when it can't be
  downloaded; otherwise set `evaluation.baseline` to `null` in test configs.
- Metric names follow `{family}_{phase}/{name}` (`pick_P/mae_s`,
  `det_S/f1`), and `det/f1_mean` is the mean of the P and S F1. Baseline
  metrics use the prefix `baseline/`, so they form one dashboard section.
  Test metrics are `test_…` and `baseline/test_…`. Keep this scheme; the
  CLI table and `training.monitor` (a `Literal` of these names; its
  min/max direction follows from the name) rely on it.
- Do not log values every epoch that cannot change between epochs. The
  baseline (fixed weights, deterministic validation windows) and the
  counts are logged once.
- MLflow 3's UI treats experiments without the `mlflow.experimentKind` tag
  as GenAI. Create experiments with `tracking.run.set_experiment()`, which
  tags them `custom_model_development`.
- Tests: `uv run pytest`. Use a small SeisBench dataset, or a synthetic
  in-memory `WaveformDataset`, with a CPU-only short run. Tests must not
  download large datasets.
- MLflow: pin `mlflow>=3`. The code reads only `MLFLOW_TRACKING_URI`, so it
  works with any backend. Since MLflow 3.7 the default local backend is
  `sqlite:///mlflow.db`. MLflow 3.18 refuses the `./mlruns` file store unless
  `MLFLOW_ALLOW_FILE_STORE=true` is set. The cluster uses that folder store
  on the shared filesystem (decision D-1 in `docs/requirements.md`). Never
  use SQLite on the shared filesystem.
- `mlflow.pytorch.log_model` uses the `pt2` format by default in 3.18. A
  `pt2` model saved from CPU weights cannot be loaded on a GPU node, so we
  log with `serialization_format="pickle"` (plus an `input_example`).
- The cluster uses conda. Do not assume `uv` in the Slurm scripts.

## On the GFZ cluster

- All GPU nodes of `qdefault` sit in the reservation `GPU`. A GPU job
  without `--reservation=GPU` never starts, and squeue shows the misleading
  reason `Priority`. Submit with
  `sbatch --reservation=GPU slurm/train.sbatch <config>`.
- The conda env has an editable install, so jobs run the code of the main
  checkout as it is when they start (and import some modules late, e.g. at
  the end of training). Do not edit, merge or switch branches in the main
  checkout while jobs are queued or running; use a git worktree instead.
- The version (setuptools-scm) is fixed at install time. After every
  commit, run `pip install -e . --no-deps` in the conda env, or runs log a
  stale version. Commit before launching runs: a dirty tree gives a
  `.dYYYYMMDD` version that does not identify the code.
- Compute nodes have no git executable; the package version carries the
  commit. MLflow's "Failed to import Git" warnings in the job logs are
  harmless.
- pytest is not in the conda env. Run the tests against it with
  `uv run --no-project --python <conda env>/bin/python --with pytest python -m pytest`.
  There, `tests/test_hpo.py::test_two_workers_share_one_study` fails with
  `FileNotFoundError .../bin/seisbench-trainer` (it looks for the console
  script next to the interpreter); that is the environment, not the code.
  To test a worktree, put its `src` first with `PYTHONPATH`, otherwise the
  editable install of the main checkout is imported.
- The cluster's `uv` (0.6) rewrites the whole `uv.lock` in an older
  format. Do not run `uv lock` there; update the lock on the laptop.
- The Rich progress bar does not reach the Slurm log while a run is going.
  Follow progress in the MLflow metrics (one `det/f1_mean` row per epoch).
- With pandas 3, `Series.to_numpy()` can return read-only arrays; combine
  masks with `a = a | b`, not `a |= b`.
- Experiment findings that set defaults are in `docs/experiments.md`.
  Add new ones there, with the MLflow experiment and the commit.

## Working rules

- Write the simplest code that solves the current task. Do not add
  speculative abstractions, plugin systems or unrequested options.
- Keep changes surgical. Do not reformat or refactor untouched code.
- Prefer native SeisBench, Lightning and MLflow features over custom
  re-implementations. Examples: `ModelCheckpoint`, MLflow system metrics, the
  MLflow model registry.
  Exception: the training perturbations (`data/perturbations.py`) are
  reviewed re-implementations of `seisbench.generate` augmentations, because
  the originals mix components, label erased picks or fail mid-training.
  Each class documents its deviations; keep them when you add one.
- Each run must be reproducible from its MLflow record alone: the resolved
  config, the seed, the dataset name and version, the seisbench-trainer version
  (which names the git commit), and the
  package versions.
- Before you report a change as done, run the linter and the tests.
- Do not commit data, checkpoints, `mlflow.db`, `mlruns/` or `mlartifacts/`.
