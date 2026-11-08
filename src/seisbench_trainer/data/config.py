"""Data and augmentation config."""

from typing import Annotated, Literal, Self

from pydantic import (
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from seisbench_trainer.data.sources import DatasetConfig, SeisBenchDataset
from seisbench_trainer.schema import StrictModel

ComponentOrder = Annotated[str, StringConstraints(pattern=r"^[A-Z0-9]{1,3}$")]


def _default_datasets() -> list[DatasetConfig]:
    return [SeisBenchDataset(name="ETHZ")]


class DataConfig(StrictModel):
    """Datasets to load and how to load them."""

    datasets: list[DatasetConfig] = Field(
        default_factory=_default_datasets,
        min_length=1,
        description="One or more datasets. Their splits are concatenated.",
    )
    component_order: ComponentOrder = Field(
        default="ZNE",
        description="Waveform component order the model is trained on.",
    )
    sampling_rate: PositiveFloat = Field(default=100.0, description="In Hz.")
    batch_size: PositiveInt = 1024
    num_workers: PositiveInt = Field(
        default=8, description="DataLoader workers. Must be at least 1."
    )

    @field_validator("component_order")
    @classmethod
    def _unique_components(cls, value: str) -> str:
        if len(set(value)) != len(value):
            raise ValueError(f"components must be unique, got {value!r}")
        return value


class AugmentationConfig(StrictModel):
    """Training augmentations and labels. Durations in seconds."""

    pick_context_probability: float = Field(
        default=2 / 3,
        ge=0.0,
        le=1.0,
        description="Probability of first cutting a context window around a "
        "random pick, before the random crop to the model input length.",
    )
    pick_context_s: PositiveFloat = 60.0
    pick_context_before_s: NonNegativeFloat = 30.0
    window_s: PositiveFloat | None = Field(
        default=None,
        description="Model input length. None uses the model's own input length.",
    )
    normalization: Literal["peak", "std"] = "peak"
    label_sigma_s: PositiveFloat = Field(
        default=0.1, description="Width of the pick label curve."
    )
    label_shape: Literal["gaussian", "triangle", "box"] = "gaussian"

    @model_validator(mode="after")
    def _context_contains_pick(self) -> Self:
        if self.pick_context_before_s > self.pick_context_s:
            raise ValueError("pick_context_before_s must not exceed pick_context_s")
        return self
