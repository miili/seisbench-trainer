---
icon: lucide/house
hide:
  - navigation
  - toc
---

<div class="sbt-hero" markdown>

<div class="sbt-hero__text" markdown>

<p class="sbt-eyebrow">seisbench-trainer</p>

# Train SeisBench pickers from one JSON file { .sbt-title }

Fine-tune or train phase pickers on any
[SeisBench](https://github.com/seisbench/seisbench) or local dataset, or
several at once, on a laptop or a GPU cluster. Good defaults, checked
configs, every run tracked in [MLflow](https://mlflow.org) and exported as
a ready-to-use SeisBench model.

[:lucide-rocket: Quickstart](quickstart.md){ .md-button .md-button--primary }
[:lucide-book-open: Guide](guide/index.md){ .md-button }

<p class="sbt-note">PhaseNet today; other pickers plug in as new model types.</p>

</div>

![PhaseNet as a U-Net: a three-component seismogram with two overlapping events goes down the encoder and up the decoder, and comes out as P, S and noise probabilities that peak at every arrival](assets/images/hero.svg){ .sbt-hero__image }

</div>

## Why seisbench-trainer

<div class="grid cards sbt-cards" markdown>

-   :lucide-tag:{ .lg .middle } __Better labels__

    ---

    Every catalogued event in a window is labelled, not taught as noise,
    and a phase the analyst did not read is masked, not punished.

    **+17 %** located events in a qseek swarm benchmark, **+0.012** test F1
    { .sbt-proof }

    [:octicons-arrow-right-24: Labels](guide/labels.md)

-   :lucide-chart-line:{ .lg .middle } __Seismological insight in MLflow__

    ---

    Not just a loss curve: pick residuals and detection F1 per phase every
    epoch, next to the pretrained model, on clean and noisy windows, with
    figures of residuals, detection scores and ROC.

    `pick_P/mae_s`, `det_S/f1`, `noisy/det/f1_mean` side by side across
    runs
    { .sbt-proof }

    [:octicons-arrow-right-24: Tracking and models](guide/tracking.md)

-   :lucide-fingerprint:{ .lg .middle } __Reproducible by default__

    ---

    Every run records its resolved config, seed, datasets, package
    versions and the git commit; resplits and validation windows are
    seeded.

    Any result can be rerun from its MLflow record alone
    { .sbt-proof }

    [:octicons-arrow-right-24: Training](guide/training.md#reproducibility)

</div>

## How it works

Six building blocks in three stages, each labelled with its config section:

![The building blocks: config and data; labels, model, training and validation, repeated by HPO; the SeisBench model; MLflow records every block](assets/images/workflow.svg){ .sbt-figure .sbt-wide }

## What you get

Every run logs a figure like this per phase and epoch: pick residuals,
detection scores against the threshold, ROC, and a summary next to the
pretrained baseline. Here PhaseNet trained on BohemiaSaxony picks 94.5 % of
the P arrivals within 0.1 s, the pretrained model 78.6 %.

![P-phase validation figure: the trained model against the pretrained baseline](assets/images/validation-P.png){ .sbt-figure .sbt-figure--light }

## Try it

```bash
git clone https://github.com/miili/seisbench-trainer && cd seisbench-trainer
uv sync
uv run seisbench-trainer train configs/synthetic-smoke.json   # about a minute on a CPU
uv run mlflow ui
```

[:lucide-rocket: Start with the quickstart](quickstart.md){ .md-button .md-button--primary }
[:lucide-lightbulb: See the findings](findings/index.md){ .md-button }
