"""Optimizer and learning-rate scheduler configs.

Each config builds its torch object. torch is imported only when building.
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, NonNegativeFloat, NonNegativeInt, PositiveFloat

from seisbench_trainer.schema import StrictModel

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import LRSchedulerConfigType
    from torch.nn import Parameter
    from torch.optim import Optimizer

Mode = Literal["min", "max"]


class AdamConfig(StrictModel):
    type: Literal["adam"] = "adam"
    lr: PositiveFloat = 1e-3
    weight_decay: NonNegativeFloat = 0.0

    def build(self, params: Iterator["Parameter"]) -> "Optimizer":
        from torch.optim import Adam

        return Adam(params, lr=self.lr, weight_decay=self.weight_decay)


class AdamWConfig(StrictModel):
    type: Literal["adamw"] = "adamw"
    lr: PositiveFloat = 1e-3
    weight_decay: NonNegativeFloat = 1e-2

    def build(self, params: Iterator["Parameter"]) -> "Optimizer":
        from torch.optim import AdamW

        return AdamW(params, lr=self.lr, weight_decay=self.weight_decay)


class CosineConfig(StrictModel):
    """Cosine annealing over all training epochs."""

    type: Literal["cosine"] = "cosine"
    eta_min: NonNegativeFloat = 0.0

    def build(
        self, optimizer: "Optimizer", epochs: int, monitor: str, mode: Mode
    ) -> "LRSchedulerConfigType":
        from torch.optim.lr_scheduler import CosineAnnealingLR

        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=self.eta_min)
        return {"scheduler": scheduler, "interval": "epoch"}


class PlateauConfig(StrictModel):
    """Reduce the learning rate when the selection metric stops improving."""

    type: Literal["plateau"] = "plateau"
    factor: float = Field(default=0.5, gt=0.0, lt=1.0)
    patience: NonNegativeInt = 5

    def build(
        self, optimizer: "Optimizer", epochs: int, monitor: str, mode: Mode
    ) -> "LRSchedulerConfigType":
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        scheduler = ReduceLROnPlateau(
            optimizer, mode=mode, factor=self.factor, patience=self.patience
        )
        return {"scheduler": scheduler, "monitor": monitor, "interval": "epoch"}


class ExponentialConfig(StrictModel):
    """Multiply the learning rate by ``decay`` after every epoch.

    AI4EPS PhaseNet decays Adam's rate (0.01) by 0.9 per epoch (staircase
    ``exponential_decay`` with one epoch of steps).
    """

    type: Literal["exponential"] = "exponential"
    decay: float = Field(
        default=0.9, gt=0.0, le=1.0, description="Learning-rate factor per epoch."
    )

    def build(
        self, optimizer: "Optimizer", epochs: int, monitor: str, mode: Mode
    ) -> "LRSchedulerConfigType":
        from torch.optim.lr_scheduler import ExponentialLR

        scheduler = ExponentialLR(optimizer, gamma=self.decay)
        return {"scheduler": scheduler, "interval": "epoch"}


OptimizerConfig = Annotated[AdamConfig | AdamWConfig, Field(discriminator="type")]
SchedulerConfig = Annotated[
    CosineConfig | PlateauConfig | ExponentialConfig, Field(discriminator="type")
]
