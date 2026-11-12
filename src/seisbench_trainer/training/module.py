"""LightningModule for any picker model config."""

from __future__ import annotations

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor

from seisbench_trainer.config import TrainConfig
from seisbench_trainer.data.loaders import Batch
from seisbench_trainer.units import seconds_to_samples


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
        window_s = self.config.augmentation.window_s
        if window_s is None:
            return self.config.model.window_samples(self.model)
        return seconds_to_samples(window_s, self.config.data.sampling_rate)

    def forward(self, x: Tensor) -> Tensor:
        return self.config.model.forward(self.model, x)

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        logits = self.config.model.forward_logits(self.model, batch["X"])
        loss = self.loss(logits, batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> dict[str, Tensor]:
        logits = self.config.model.forward_logits(self.model, batch["X"])
        loss = self.loss(logits, batch)
        self.log("val_loss", loss, prog_bar=True)
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
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
