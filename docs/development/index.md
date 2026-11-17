---
icon: lucide/wrench
---

# Development

The rules for working on the code, for people and coding agents alike, are
in `AGENTS.md` at the repository root. The short version:

- **Configuration** is Pydantic v2, one JSON file per run, no layering.
  Every choice between alternatives is a discriminated union: each member is
  a class with a `type` tag that builds its own runtime object. To extend the
  trainer, add a member to a union; no `if`/`elif` dispatch, no registries.
- **Every config field has a `Field(description=...)`** and every config
  class a docstring. They are the [config reference](../reference/config.md)
  and the schema of `seisbench-trainer config --schema`; a test fails on an
  undocumented field.
- **Seconds become samples in one place** (`units.seconds_to_samples`, with
  `data.sampling_rate`), and output channels are mapped by label name, never
  by index.
- **Labels, masks and pick metrics read the arrival arrays** of
  `data.arrivals` (`phases.phase_arrivals`), never the pick columns.
- **Simplest code that solves the task**; surgical changes; prefer native
  SeisBench, Lightning and MLflow features.

## Layout

```text
src/seisbench_trainer/
  cli.py            Typer app (the only module that imports typer)
  console.py        Rich console, logging, tables (the only one that configures Rich)
  config.py         TrainConfig and its sections
  data/             sources, preprocessing, arrivals, labels, steps, perturbations, loaders
  models/           model configs (adapters), losses
  training/         LightningModule, optimizers, train()
  metrics/          pure NumPy pick and detection scores, figures
  tracking/         MLflow run metadata, model export, the evaluation callback
  evaluate.py       test and cross-dataset evaluation, model loading
  hpo/              Optuna search space and worker (optional extra)
```

`metrics` depends on nothing in the package; `cli` sits on top. SeisBench
and torch are imported only where data or models are loaded, so config
handling and the CLI stay fast.

## Checks

```bash
uv run pre-commit run --all-files     # ruff check, ruff format
uv run pytest                         # about 3 min on 16 cores
```

The tests use a generated dataset and never download data; tests that need
the pretrained `original` PhaseNet skip when it cannot be downloaded. On a
shared login node with few cores, run the suite as a batch job.

Before a change that could move results, compare a fixed end-to-end run
(training with every augmentation, a test evaluation and a baseline) before
and after: all metrics should match exactly, or the difference should be the
intended one.

## Documentation

The site is built with [Zensical](https://zensical.org) from `docs/` and
published to GitHub Pages by `.github/workflows/docs.yml` when a commit on
`main` is tagged with a version (`v*`, e.g. `git tag v0.2.0 && git push
origin v0.2.0`); the tag is also the package version. The site therefore
shows the documentation of the latest release, not of `main`.

```bash
uv run python scripts/gen_reference.py   # after changing config fields or CLI options
uvx zensical serve                       # preview at http://localhost:8000
```

The config and CLI reference pages are generated and committed; a test
fails when they are out of date.

## Experiments

Experiments that set a default are recorded in the
[experiment log](../experiments.md): the question, the MLflow experiment, the
commit, the numbers, and what was concluded. Commit before you start runs,
so the logged version names the code.

## Design history

The [requirements](../requirements.md) and the [implementation plan](../plan.md)
written at the start of the project, kept for the design decisions they
record (they are not user documentation).
