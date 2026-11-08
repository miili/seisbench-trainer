# Implementation plan: seisbench-trainer

This plan implements `docs/requirements.md` (the IDs in brackets refer to
it). We build in small steps. Each step ends in a working, tested state and
adds no code for later steps.

## Principles

- Keep it simple. The core MLflow callback is small; it gets bigger only when
  a step needs it.
- Keep it modular [NFR-1]. Each concern is a subpackage. Alternatives are
  Pydantic discriminated unions, and each union member builds its own runtime
  object [FR-C7]. Extending the code means adding a class, not editing a
  dispatch.
- The CLI uses Typer. All console output goes through Rich [FR-CLI1,
  FR-CLI1a].
- Every step is done when all of these pass: `uv run pre-commit run
  --all-files` (ruff, pyright strict), `uv run pytest`, and a laptop smoke
  run.
- Development and tests run on the laptop: CPU only, synthetic data. The
  cluster is used from step 4 onwards, to check results on real data.

## Architecture (target)

```
src/seisbench_trainer/
  __init__.py
  py.typed
  cli.py                 Typer app: config, train, evaluate, register, export, hpo
  console.py             Rich console, RichHandler logging setup, table helpers
  config.py              TrainConfig composing the sections; TrainingConfig, EvaluationConfig, TrackingConfig
  schema.py              StrictModel: shared base (extra="forbid", frozen)
  units.py               seconds_to_samples() and other unit conversions
  data/
    __init__.py          re-exports
    config.py            DataConfig, AugmentationConfig
    phases.py            PHASES, PHASE_COLUMNS, augmentation types (no heavy imports)
    sources.py           SeisBenchDataset | LocalDataset | SyntheticDataset (tag "type"); each has .load()
    synthetic.py         write_synthetic_dataset(): noise plus P and S wavelets
    augmentations.py     train and validation pipelines
    loaders.py           split, concatenate, preload, DataLoaders
  models/
    __init__.py          ModelConfig = Annotated[PhaseNetConfig | ..., Field(discriminator="type")]
    base.py              PickerModelConfig (ABC): build, label_augmentations, forward, loss, phase_probabilities
    phasenet.py          PhaseNetConfig
    eqtransformer.py     EQTransformerConfig (step 10)
  training/
    __init__.py
    optim.py             OptimizerConfig union (Adam | AdamW) and SchedulerConfig union (None | Cosine | Plateau)
    module.py            PickerModule (LightningModule); uses ModelConfig, nothing model-specific
    train.py             train(config) -> run_id
  metrics/
    __init__.py
    picks.py             pick_accuracy (window-based), detect_picks (label-free)
    scores.py            precision_recall_f1, best_threshold, roc_auc, summary
    plots.py             matplotlib figures
  tracking/
    __init__.py
    run.py               run metadata: config, git, versions, Slurm tags; LoggedModel export
    callbacks.py         EvaluationCallback (metrics and figures to MLflow)
    registry.py          register, export
  evaluate.py            test and cross-dataset evaluation
  hpo/                   imported only with the [hpo] extra
    __init__.py
    space.py             FloatParam | IntParam | CategoricalParam (tag "type")
    study.py             SamplerConfig and PrunerConfig unions; worker loop
tests/
  conftest.py            synthetic SeisBench dataset fixture (tmp_path)
  test_*.py
typings/seisbench/       partial type stubs for the seisbench API in use
configs/                 example JSON configs
slurm/                   sbatch scripts
```

Dependency direction:

- `units` and `metrics` depend on nothing in the package.
- `data`, `models` and `training.optim` depend only on `units`, `pydantic`
  and `seisbench`.
- `config` composes their config classes.
- `training`, `tracking`, `evaluate` and `hpo` put the pieces together.
- `cli` sits on top.

Nothing below `cli` imports `typer`. Only `console` configures Rich.

Example of the union pattern:

```python
class PickerModelConfig(BaseModel, ABC):
    pretrained: str | None = None

    @abstractmethod
    def build(self) -> sbm.WaveformModel: ...
    @abstractmethod
    def label_augmentations(
        self, model: sbm.WaveformModel, sigma: int
    ) -> list[Augmentation]: ...
    @abstractmethod
    def loss(self, pred: Tensor, batch: Batch) -> Tensor: ...
    @abstractmethod
    def phase_probabilities(self, pred: Tensor) -> dict[Phase, Tensor]: ...


class PhaseNetConfig(PickerModelConfig):
    type: Literal["PhaseNet"] = "PhaseNet"
    ...


ModelConfig = Annotated[PhaseNetConfig, Field(discriminator="type")]
# step 10: Annotated[PhaseNetConfig | EQTransformerConfig, Field(discriminator="type")]
```

Tested with Pydantic 2.14: a union with only one member and
`Field(discriminator=…)` works. It parses, gives a correct schema, and
rejects unknown tags with `union_tag_invalid`.

---

## Step 0: Project setup

Tasks:
- Add the dependencies to `pyproject.toml`:
  - core: `torch`, `lightning`, `seisbench`, `mlflow>=3`, `pydantic>=2`,
    `typer`, `rich`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`
    [NFR-6];
  - extra `hpo`: `optuna`, `optuna-integration`;
  - dev group: `ruff`, `pyright`, `pytest`, `pre-commit`.
- Laptop: get a CPU-only `torch` from the PyTorch CPU index through
  `[tool.uv.sources]`, so a small laptop does not download the CUDA wheels.
  The cluster's `pip install -e .` ignores `tool.uv`, so its conda env is
  not affected.
- Add ruff (`E,F,I,UP,B,SIM,ANN,RUF`) and pyright (`strict`, untyped
  packages handled one by one) to `pyproject.toml`. Add
  `.pre-commit-config.yaml` [NFR-8, NFR-9] and `py.typed`.
- Add `.gitignore` entries: `mlruns/`, `mlflow.db`, `mlartifacts/`,
  `checkpoints/`, `*.ckpt`, `lightning_logs/`.
- Create the subpackage skeleton, and add `console.py` (Rich console and
  `RichHandler`). The Typer app in `cli.py` starts with a `version`
  command; the entry point is `seisbench-trainer = "seisbench_trainer.cli:app"`.

Done when: `uv sync`, `pre-commit run --all-files` and `pytest` all pass,
and `uv run seisbench-trainer --help` renders through Rich.

## Step 1: Configuration [FR-C1–C7]

Tasks:
- Config models with `extra="forbid"`, using discriminated unions [FR-C7].
  Each subpackage owns its own config classes:
  - `data`: `SeisBenchDataset | LocalDataset` (tag `type`), and
    `DataConfig` (`datasets: list[DatasetConfig]`, `component_order`,
    `sampling_rate`, `batch_size`, `num_workers`);
  - `models`: `ModelConfig` (only `PhaseNetConfig` for now);
  - `training.optim`: `OptimizerConfig` and `SchedulerConfig`;
  - `AugmentationConfig`: windows and label sigma in seconds;
  - `EvaluationConfig`: tolerances, pick window, peak height and distance,
    baseline, `plot_every_n_epochs`;
  - `TrackingConfig`: experiment name, tags;
  - `TrainingConfig`: epochs, seed, accelerator, precision, selection
    metric, optimizer, scheduler.
- `config.py` composes these into `TrainConfig`. `load_config(path)` and
  `dump_config()` use JSON only.
- `units.seconds_to_samples()` is the only place that converts seconds to
  samples [FR-C6].
- CLI: `seisbench-trainer config [--schema]` prints the default config or
  its JSON schema, with Rich JSON highlighting.
- Tests:
  - the default config survives a dump and reload unchanged;
  - every union member parses from JSON and selects the right class;
  - an unknown tag, an unknown key, `num_workers=0` and an invalid component
    order are each rejected with a readable error.

The `hpo` section is added in step 9. Model-dependent checks (for example,
whether the window length fits the model) need the built model, so they come
in step 3.

Status: done. Every union is tagged by a `type` field, including the model
(`"model": {"type": "PhaseNet", "pretrained": "original"}`). The model union
has only one member for now; the abstract base and `build()` come in
step 3.

## Step 2: Data [FR-D1–D5]

Tasks:
- `tests/conftest.py`: a tiny synthetic SeisBench dataset written with
  `WaveformDataWriter`. It has a few hundred traces with P and S picks plus
  noise traces, and train/dev/test splits. This fixture is used by every
  later test.
- `data/sources.py`: `SeisBenchDataset.load()` (a class from
  `seisbench.data`) and `LocalDataset.load()` (a path), each applying its
  metadata filters.
- `data/loaders.py`:
  - load each dataset with the run's component order and sampling rate,
    split it, concatenate the splits per type, and call
    `preload_waveforms()` [FR-D3];
  - build DataLoaders with persistent workers [FR-D4].
- `data/augmentations.py`: port the reference pipeline (`WindowAroundSample`,
  `RandomWindow`, `Normalize`, `ChangeDtype`). The label augmentations come
  from the model config. The validation windows use a random offset with a
  fixed seed [FR-E2b]. Pass the true pick samples through as an extra batch
  key [FR-E2a.3].
- Tests: batch shapes and dtypes; the label peaks agree with the pick-sample
  key; two passes over the validation set give identical windows.

Status: done.

- The synthetic data is part of the package: `data/synthetic.py`, used
  through the `{"type": "synthetic"}` dataset source. It is written once per
  set of parameters into the system temp folder.
- `pick_samples` is a `(phases, 4)` float array, NaN-padded, in the order of
  `PHASES` (P, S).
- seisbench is typed through partial stubs in `typings/seisbench/`, and is
  imported only when data is loaded.

## Step 3: Model config and LightningModule [FR-M1–M5, FR-T1]

Tasks:
- `models/base.py`: the abstract `PickerModelConfig`, as in the
  architecture example above.
- `models/phasenet.py`: `PhaseNetConfig`, porting the reference loss and the
  `ProbabilisticLabeller`.
- `training/optim.py`: each union member builds its own optimizer or
  scheduler, for example `AdamConfig.build(params) -> torch.optim.Adam`.
- `training/module.py`: `PickerModule(LightningModule)` takes a
  `ModelConfig` and calls only the base-class interface. It gets the label
  and component order from the built model [FR-M4].
- Tests: one forward and backward pass on CPU on the synthetic data; a
  pretrained `original` PhaseNet loads, and its component order is read
  correctly. Mark tests that need a download as skippable when offline.

Status: done.

- The base class has one more method than planned:
  `forward(model, x) -> Tensor`. It returns probabilities as
  (batch, channel, sample), so `loss` and `phase_probabilities` work on one
  layout for every model family (PhaseNet calls `annotate_batch_pre` first,
  as in the reference code).
- A pretrained model must be fine-tuned on data in its own component
  order and at its own sampling rate. A mismatch is an error, and the message
  names the value to set (for example `original` needs `ENZ`). Feeding
  another order would swap the input channels that the weights expect.
- Shared constants (`PHASES`, `PHASE_COLUMNS`, `Augmentation`) live in
  `data/phases.py`, which has no heavy imports.
- DataLoaders start their workers with `fork` explicitly, so the workers
  share the preloaded waveforms.

## Step 4: Training with minimal MLflow [FR-T2–T5, FR-L1–L4, FR-L6, FR-L7]

This is the first end-to-end milestone.

Tasks:
- `training/train.py`: `train(config) -> run_id` does the following:
  - `seed_everything`;
  - sets up the data, the module and a `lightning` `Trainer` with
    `accelerator="auto"` and `RichProgressBar`;
  - uses `MLFlowLogger` for the scalar metrics. Check that it works with
    MLflow 3.x, and turn on async logging;
  - uses `ModelCheckpoint` on the selection metric to a local or scratch
    directory, with nothing uploaded [FR-L6].
- `tracking/run.py`:
  - run metadata: `config.json`, flattened params, the git commit and dirty
    flag, package versions, host and the Slurm IDs as tags [FR-C5, FR-L2];
    system metrics are enabled once [FR-L3];
  - at the end, load the best checkpoint, export it as a SeisBench
    `.json`/`.pt` pair, check the component order [FR-M5], and log it as an
    MLflow LoggedModel (with an `input_example`, required by the default
    `pt2` format) with the SeisBench files attached [FR-L4].
- CLI: `seisbench-trainer train config.json`. It prints a Rich summary table
  of the final metrics and the run ID.
- Tests: a two-epoch CPU run on synthetic data into a `tmp_path` SQLite
  store. Check that the run is `FINISHED`, has `train_loss` and `val_loss`
  for each epoch, has `config.json`, and has a LoggedModel whose SeisBench
  files load with `sbm.PhaseNet.load()`.
- Cluster: a conda env (Python 3.13); `slurm/train.sbatch` [FR-CLI2]; the
  folder store with `MLFLOW_ALLOW_FILE_STORE=true` [D-1]; one fine-tuning
  run on BohemiaSaxony. Check the time per epoch and GPU use [NFR-4], and
  `df -T` on the shared filesystem.

## Step 5: Evaluation metrics and the MLflow callback [FR-E1–E4, FR-E6]

Tasks:
- `metrics/picks.py` and `metrics/scores.py`: pure, typed and vectorised
  where possible:
  - `pick_accuracy()` (window-based) with all the FR-E2a fixes;
  - `detect_picks()` (label-free peak matching);
  - `precision_recall_f1()`, `best_threshold()`, `roc_auc()`;
  - `summary()`: residual statistics including percentiles, MAD and the
    fraction within each tolerance.

  Results are frozen dataclasses or NamedTuples. The functions work on phase
  names, never on fixed channel indices.
- Noise examples for the negative class [FR-E2c]: the dataset's noise
  traces, plus the count of negatives used.
- `metrics/plots.py`: port the four figures from the reference code.
- `tracking/callbacks.py`: one `EvaluationCallback` that:
  - collects predictions through `ModelConfig.phase_probabilities()`;
  - computes and logs the `pick_*` and `det_*` metrics in
    `on_validation_epoch_end`;
  - logs images with `step=epoch` every *n* epochs;
  - adds an optional baseline model, with its inputs reordered to its own
    component order [FR-E3].
- Tests: known-answer tests for every metric function. They use
  hand-built arrays covering each case: a pick at the window edge, two picks
  in one window, no picks, a perfect prediction and a shifted prediction.
  A regression test for the old edge-mask bug.
- Cluster check: on BohemiaSaxony, the validation F1 and MAE are close to
  the thesis values.

## Step 6: Test and cross-dataset evaluation [FR-E5, FR-E5a]

Tasks:
- `evaluate.py`: `evaluate(model_ref, datasets, split, tolerances,
  thresholds)`. `model_ref` is `models:/…`, a run ID, a SeisBench
  pretrained name, or a local path. It logs the metrics linked to `model_id`
  and the dataset, plus a `log_table` summary.
- Optional `evaluation.test_at_end` runs it right after training, with
  the thresholds selected on the validation set.
- CLI: `seisbench-trainer evaluate --model … --dataset … --split test`,
  which prints a Rich results table.
- Tests: evaluating the step-4 model on the synthetic test split logs a
  table and metrics linked to the model.

## Step 7: Combined datasets [FR-D2a, FR-D2b]

Tasks:
- Validation and test metrics are logged for each dataset (a `ds=<name>`
  infix in the metric key) and for the combined set.
- An optional per-dataset `weight` field on the dataset configs,
  implemented with a `WeightedRandomSampler` on the training loader.
- Tests: two synthetic datasets give per-dataset metrics; the weights
  change the share each dataset contributes to the training batches.

## Step 8: Registry, export and batch runs [FR-L5, FR-L8, FR-CLI1, FR-CLI3]

Tasks:
- `tracking/registry.py` with two CLI commands:
  - `register <run_id> --alias champion`;
  - `export models:/<name>@<alias> <dir>`, which writes the SeisBench
    `.json`/`.pt` files for Qseek.
- Naming defaults: experiment `<datasets>/<model>`; run name
  `<variant>-<config hash>`; registered name
  `<model>-<datasets>[-from-<pretrained>]`.
- `slurm/train_array.sbatch`: reads line `$SLURM_ARRAY_TASK_ID` of a text
  file of config paths.
- Tests: register, then export, then `sbm.PhaseNet.load()` succeeds on the
  exported files.

## Step 9: HPO, optional [FR-H1–H7]

Tasks:
- `hpo/space.py`: `FloatParam | IntParam | CategoricalParam` (tag `type`).
  Each has a `path` (a dotted path into `TrainConfig`) and a
  `.suggest(trial)` method.
- `hpo/study.py`:
  - `SamplerConfig` union (TPE | Random) and `PrunerConfig` union
    (None | Median | Hyperband), each with `.build()`;
  - storage is `JournalStorage` with a file backend on the shared
    filesystem; the lock type depends on `df -T`. Check the API for Optuna
    5.x;
  - the `HPOConfig` section is added to `TrainConfig` as an optional field.
- `optuna` is imported only inside `hpo/`. The CLI command imports it lazily
  and prints a Rich error with the install hint when the extra is missing
  [FR-H1].
- Each trial applies its sampled values to a copy of the config and
  validates the result again. It then calls the same `train()` function
  inside an MLflow child run under the study's parent run, reusing the data
  that is already preloaded [FR-H4]. A pruning callback uses the selection
  metric [FR-H5].
- The parent run logs the best parameters and the Optuna plots [FR-H6].
  The CLI prints a Rich table of the top trials.
- CLI: `seisbench-trainer hpo config.json`. Slurm:
  `slurm/hpo_worker.sbatch`, an array of workers sharing one study
  [FR-H3, FR-H7].
- Tests (skipped without the extra): a three-trial study on synthetic data,
  run as two worker processes, gives one parent run and three child runs.

## Step 10: EQTransformer [FR-M1]

Tasks:
- `models/eqtransformer.py`: `EQTransformerConfig(PickerModelConfig)`, with
  a detection label, a weighted multi-output loss, and the tuple-output
  mapping. The window length comes from `model.in_samples` (6000).
  Add it to the `ModelConfig` union.
- Changes are allowed only in `models/` and the tests. If other modules
  need changing, the base-class interface is wrong: fix the interface, not
  the callers.
- Tests: the same smoke test as step 4 with `"model": {"type": "EQTransformer"}`.

---

## Milestones

| Milestone | After step | Result |
| --- | --- | --- |
| M1 | 4 | A PhaseNet run trains on the cluster and is tracked in MLflow with an exported SeisBench model |
| M2 | 6 | The thesis evaluation is reproduced (per-epoch metrics, test results, comparison with the baseline) |
| M3 | 8 | Many datasets and many models are trained, registered and exported routinely |
| M4 | 9 | Distributed HPO |
| M5 | 10 | A second model family |

## Risks and checks

- **Lightning `MLFlowLogger` with MLflow 3.x:** check it in step 4. If it
  falls short, log the scalars from our own callback with the MLflow API.
- **pyright strict with untyped seisbench:** expect warnings for unknown
  types at that boundary. Keep the seisbench calls inside the `build()` and
  `load()` methods and a few typed wrapper functions, so the rest of the code
  stays clean.
- **Optuna 5.0 is new:** check the journal storage and pruning-callback
  APIs before step 9.
- **Folder-store opt-out:** pin the MLflow version range in the conda env
  (D-1). Watch the release notes for removal of the opt-out.
