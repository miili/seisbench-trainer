"""Per-epoch validation metrics and figures."""

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
from seisbench_trainer.metrics.plots import PhaseResult, figure_to_array, phase_figure
from seisbench_trainer.metrics.scores import summarize
from seisbench_trainer.models import ModelConfig
from seisbench_trainer.training.module import PickerModule
from seisbench_trainer.units import seconds_to_samples

BASELINE = "baseline_"


class EvaluationCallback(pl.Callback):
    """Pick-accuracy (``pick_*``) and detection (``det_*``) metrics per epoch.

    Metrics are logged through the LightningModule, so they are available as
    selection metric for checkpointing, early stopping and HPO. Figures are
    logged to MLflow as step-indexed images. The optional baseline model gets
    the prefix ``baseline_``.
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

        if self.baseline is not None:
            config, model, reorder = self.baseline
            model.to(x.device)
            with torch.no_grad():
                baseline_pred = config.forward(model, x[:, reorder])
            self._collect(BASELINE, config, model, baseline_pred, pick_array)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.sanity_checking or not self.collected:
            self.collected.clear()
            return
        data = self.config.data
        tolerances = self.config.evaluation.test_tolerances_s
        merged = {
            key: PickScores.concatenate(parts) for key, parts in self.collected.items()
        }
        self.collected.clear()

        metrics: dict[str, float] = {}
        for key, scores in merged.items():
            for name, value in summarize(
                scores, data.sampling_rate, tolerances
            ).items():
                metrics[f"{key}/{name}"] = value
        for prefix in ("", BASELINE) if self.baseline else ("",):
            f1 = [metrics[f"{prefix}det_{phase}/f1"] for phase in PHASES]
            metrics[f"{prefix}det/f1_mean"] = float(np.mean(f1))
        pl_module.log_dict(metrics)

        every = self.config.evaluation.plot_every_n_epochs
        if every and trainer.current_epoch % every == 0:
            self._log_figures(trainer, merged)

    def _log_figures(self, trainer: pl.Trainer, merged: dict[str, PickScores]) -> None:
        mlflow_logger = next(
            (lg for lg in trainer.loggers if isinstance(lg, MLFlowLogger)), None
        )
        if mlflow_logger is None or mlflow_logger.run_id is None:
            return
        names = {"": self.config.model.type}
        if self.baseline is not None:
            names[BASELINE] = f"baseline {self.baseline[0].variant()}"
        epoch = trainer.current_epoch
        for phase in PHASES:
            results = {
                name: PhaseResult(
                    pick=merged[f"{prefix}pick_{phase}"],
                    detection=merged[f"{prefix}det_{phase}"],
                )
                for prefix, name in names.items()
            }
            figure = phase_figure(
                f"{phase} phase, epoch {epoch}",
                results,
                self.config.data.sampling_rate,
                self.config.evaluation.pick_window_s,
            )
            # The logger was created with mlflow.get_tracking_uri(); so is this.
            MlflowClient().log_image(  # pyright: ignore[reportUnknownMemberType]
                mlflow_logger.run_id,
                figure_to_array(figure),
                key=f"evaluation_{phase}",
                step=epoch,
            )
