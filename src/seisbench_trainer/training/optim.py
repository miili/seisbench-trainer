"""Optimizer and learning-rate scheduler configs."""

from typing import Annotated, Literal

from pydantic import Field, NonNegativeFloat, NonNegativeInt, PositiveFloat

from seisbench_trainer.schema import StrictModel


class AdamConfig(StrictModel):
    type: Literal["adam"] = "adam"
    lr: PositiveFloat = 1e-3
    weight_decay: NonNegativeFloat = 0.0


class AdamWConfig(StrictModel):
    type: Literal["adamw"] = "adamw"
    lr: PositiveFloat = 1e-3
    weight_decay: NonNegativeFloat = 1e-2


class CosineConfig(StrictModel):
    """Cosine annealing over all training epochs."""

    type: Literal["cosine"] = "cosine"
    eta_min: NonNegativeFloat = 0.0


class PlateauConfig(StrictModel):
    """Reduce the learning rate when the selection metric stops improving."""

    type: Literal["plateau"] = "plateau"
    factor: float = Field(default=0.5, gt=0.0, lt=1.0)
    patience: NonNegativeInt = 5


OptimizerConfig = Annotated[AdamConfig | AdamWConfig, Field(discriminator="type")]
SchedulerConfig = Annotated[CosineConfig | PlateauConfig, Field(discriminator="type")]
