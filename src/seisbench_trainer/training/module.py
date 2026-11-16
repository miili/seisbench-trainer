"""LightningModule for any picker model config."""

from __future__ import annotations

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor

from seisbench_trainer.config import TrainConfig
from seisbench_trainer.data.loaders import Batch
from seisbench_trainer.metrics.names import NOISY


class PickerModule(pl.LightningModule):
    """Training and validation steps. Model specifics come from the model config."""

    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self.config = config
        self.model = config.model.build(
            config.data.component_order, config.data.sampling_rate
        )
        self.loss = config.model.build_loss(self.model)

    @property
    def window_samples(self) -> int:
        """Input window length: the config override or the model default."""
        return self.config.augmentation.window_samples(
            self.config.model.window_samples(self.model), self.config.data.sampling_rate
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.config.model.forward(self.model, x)

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        logits = self.config.model.forward_logits(self.model, batch["X"])
        loss = self.loss(logits, batch)
        if not torch.isfinite(loss):
            # a NaN step makes every weight NaN; the run would go on and
            # export a model that predicts nothing
            raise FloatingPointError(
                f"training loss is {loss.item()} at step {self.global_step}"
            )
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Batch, batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        logits = self.config.model.forward_logits(self.model, batch["X"])
        loss = self.loss(logits, batch)
        clean = dataloader_idx == 0
        self.log(
            "val_loss" if clean else f"{NOISY}val_loss",
            loss,
            prog_bar=clean,
            add_dataloader_idx=False,
        )
        return {"loss": loss, "pred": self.config.model.activate(logits)}

    def configure_optimizers(self) -> OptimizerLRScheduler:
        training = self.config.training
        optimizer = training.optimizer.build(self.parameters())
        if training.scheduler is None:
            return optimizer
        scheduler = training.scheduler.build(
            optimizer,
            epochs=training.epochs,
            monitor=training.monitor,
            mode=training.monitor_mode,
            min_delta=training.min_delta,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
