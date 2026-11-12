"""Per-epoch validation metrics and figures."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any, cast

import lightning.pytorch as pl
import numpy as np
import seisbench.models as sbm
import torch
from lightning.pytorch.loggers import MLFlowLogger
from mlflow import MlflowClient
from numpy.typing import NDArray
from torch import Tensor

from seisbench_trainer.config import TrainConfig
from seisbench_trainer.data.phases import PHASES
from seisbench_trainer.metrics.picks import PickScores, detect_picks, pick_accuracy
from seisbench_trainer.metrics.plots import (
    PhaseResult,
    figure_to_png,
    phase_figure,
    slider_page,
)
from seisbench_trainer.metrics.scores import summarize
from seisbench_trainer.models import ModelConfig
from seisbench_trainer.tracking.run import dataset_label
from seisbench_trainer.training.module import PickerModule
from seisbench_trainer.units import seconds_to_samples

BASELINE = "baseline/"
"""Prefix of baseline metrics: one dashboard section for all of them."""


class EvaluationCallback(pl.Callback):
    """Pick-accuracy (``pick_*``) and detection (``det_*``) metrics per epoch.

    Metrics are logged through the LightningModule, so they are available as
    selection metric for checkpointing, early stopping and HPO. The figures
    of all epochs go into one HTML page per phase with an epoch slider
    (``evaluation/validation_<phase>.html``).

    The optional baseline has fixed weights and the validation windows are
    deterministic, so it is evaluated once, in the first epoch: its metrics
    (prefix ``baseline/``) are logged once and its scores are reused for the
    figures. Counts that do not change between epochs are logged once too.
    """

    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        evaluation = config.evaluation
        sampling_rate = config.data.sampling_rate
        self.config = config
        self.half_window = seconds_to_samples(evaluation.pick_window_s, sampling_rate)
        self.edge = seconds_to_samples(evaluation.edge_mask_s, sampling_rate)
        self.tolerance = seconds_to_samples(evaluation.tolerance_s, sampling_rate)
        self.min_distance = seconds_to_samples(
            evaluation.min_peak_distance_s, sampling_rate
        )
        self.baseline: tuple[ModelConfig, sbm.WaveformModel, list[int]] | None = None
        if evaluation.baseline is not None:
            self.baseline = self._load_baseline(evaluation.baseline)
        self.collected: dict[str, list[PickScores]] = defaultdict(list)
        self.baseline_scores: dict[str, PickScores] | None = None
        self.first_epoch_logged = False
        self.frames: dict[str, list[tuple[str, bytes]]] = defaultdict(list)
        self.figures: dict[int, dict[str, bytes]] = defaultdict(dict)
        """PNG figure per epoch and phase; the best epoch's go with the model."""
        self.thresholds: dict[int, dict[str, float]] = {}
        """F1-optimal detection threshold per epoch and phase (validation set)."""

    def _load_baseline(
        self, baseline: ModelConfig
    ) -> tuple[ModelConfig, sbm.WaveformModel, list[int]]:
        model = baseline.load_pretrained()
        data = self.config.data
        if model.sampling_rate != data.sampling_rate:
            raise ValueError(
                f"baseline expects {model.sampling_rate} Hz, "
                f"the data has {data.sampling_rate} Hz"
            )
        missing = set(model.component_order) - set(data.component_order)
        if missing:
            raise ValueError(f"baseline needs components {missing} the data lacks")
        reorder = [data.component_order.index(c) for c in model.component_order]
        model.eval().requires_grad_(False)
        return baseline, model, reorder

    def _collect(
        self,
        prefix: str,
        model_config: ModelConfig,
        model: sbm.WaveformModel,
        pred: Tensor,
        picks: NDArray[np.float32],
    ) -> None:
        evaluation = self.config.evaluation
        curves = model_config.phase_probabilities(model, pred)
        for phase_index, phase in enumerate(PHASES):
            probs = curves[phase].detach().float().cpu().numpy()
            phase_picks = picks[:, phase_index]
            self.collected[f"{prefix}pick_{phase}"].append(
                pick_accuracy(probs, phase_picks, self.half_window, self.edge)
            )
            self.collected[f"{prefix}det_{phase}"].append(
                detect_picks(
                    probs,
                    phase_picks,
                    evaluation.min_peak_height,
                    self.min_distance,
                    self.tolerance,
                )
            )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: object,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.sanity_checking:
            return
        assert isinstance(pl_module, PickerModule)
        assert isinstance(outputs, Mapping) and isinstance(batch, Mapping)
        pred = outputs["pred"]
        fields = cast(Mapping[str, object], batch)
        x = fields["X"]
        picks = fields["pick_samples"]
        assert isinstance(pred, Tensor)
        assert isinstance(x, Tensor) and isinstance(picks, Tensor)
        pick_array = picks.cpu().numpy()
        self._collect("", self.config.model, pl_module.model, pred, pick_array)

        if self.baseline is not None and self.baseline_scores is None:
            config, model, reorder = self.baseline
            model.to(x.device)
            with torch.no_grad():
                baseline_pred = config.forward(model, x[:, reorder])
            self._collect(BASELINE, config, model, baseline_pred, pick_array)

    def _metrics(
        self, scores: dict[str, PickScores], include_static: bool
    ) -> dict[str, float]:
        evaluation = self.config.evaluation
        pick_tolerances = evaluation.test_tolerances_s
        # detection only matches picks within tolerance_s: wider fractions are 1
        det_tolerances = [t for t in pick_tolerances if t < evaluation.tolerance_s]
        metrics: dict[str, float] = {}
        prefixes: set[str] = set()
        for key, value in scores.items():
            prefix = BASELINE if key.startswith(BASELINE) else ""
            family = key.removeprefix(prefix).split("_")[0]
            prefixes.add(prefix)
            tolerances = det_tolerances if family == "det" else pick_tolerances
            summary = summarize(value, self.config.data.sampling_rate, tolerances)
            for name, number in summary.items():
                static = name == "n_picks" or (
                    family == "pick" and name == "n_negatives"
                )
                if include_static or not static:
                    metrics[f"{key}/{name}"] = number
        for prefix in prefixes:
            f1 = [metrics[f"{prefix}det_{phase}/f1"] for phase in PHASES]
            metrics[f"{prefix}det/f1_mean"] = float(np.mean(f1))
        return metrics

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.sanity_checking or not self.collected:
            self.collected.clear()
            return
        merged = {
            key: PickScores.concatenate(parts) for key, parts in self.collected.items()
        }
        self.collected.clear()
        trained = {k: v for k, v in merged.items() if not k.startswith(BASELINE)}

        first = not self.first_epoch_logged
        metrics = self._metrics(trained, include_static=first)
        if self.baseline is not None and self.baseline_scores is None:
            self.baseline_scores = {
                k: v for k, v in merged.items() if k.startswith(BASELINE)
            }
            metrics |= self._metrics(self.baseline_scores, include_static=True)
        self.first_epoch_logged = True
        pl_module.log_dict(metrics)
        self.thresholds[trainer.current_epoch] = {
            phase: metrics[f"det_{phase}/threshold"] for phase in PHASES
        }

        every = self.config.evaluation.plot_every_n_epochs
        if every and trainer.current_epoch % every == 0:
            self._log_figures(trainer, trained | (self.baseline_scores or {}))

    def _log_figures(self, trainer: pl.Trainer, merged: dict[str, PickScores]) -> None:
        mlflow_logger = next(
            (lg for lg in trainer.loggers if isinstance(lg, MLFlowLogger)), None
        )
        if mlflow_logger is None or mlflow_logger.run_id is None:
            return
        model = self.config.model
        start = (
            "trained from scratch"
            if model.pretrained is None
            else f"fine-tuned from {model.weights_label()}"
        )
        names = {"": (f"{model.type}-{dataset_label(self.config)}", start)}
        if self.baseline is not None:
            baseline = self.baseline[0]
            label = f"{baseline.type} ({baseline.weights_label()})"
            names[BASELINE] = (label, "pretrained baseline")
        epoch = trainer.current_epoch
        for phase in PHASES:
            results = {
                name: PhaseResult(
                    pick=merged[f"{prefix}pick_{phase}"],
                    detection=merged[f"{prefix}det_{phase}"],
                    description=description,
                )
                for prefix, (name, description) in names.items()
            }
            evaluation = self.config.evaluation
            figure = phase_figure(
                f"{phase} phase · validation · epoch {epoch}",
                results,
                self.config.data.sampling_rate,
                evaluation.pick_window_s,
                min(evaluation.test_tolerances_s),
                evaluation.tolerance_s,
            )
            png = figure_to_png(figure)
            self.figures[epoch][phase] = png
            self.frames[phase].append((f"epoch {epoch}", png))
            # Rewritten every epoch; MLflow's own image slider shows 256 px
            # thumbnails only. The logger uses mlflow.get_tracking_uri(); so
            # does this client.
            MlflowClient().log_text(
                mlflow_logger.run_id,
                slider_page(f"Validation, {phase} phase", self.frames[phase]),
                f"evaluation/validation_{phase}.html",
            )
