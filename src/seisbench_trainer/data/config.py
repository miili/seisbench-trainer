"""Data and augmentation config."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import (
    Field,
    PositiveFloat,
    PositiveInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from seisbench_trainer.data.labels import LabelShape
from seisbench_trainer.data.sources import DatasetConfig, SeisBenchDataset
from seisbench_trainer.data.steps import (
    ChannelDropout,
    PickWindow,
    RandomWindow,
    RealNoise,
    RotateHorizontal,
    TrainStep,
)
from seisbench_trainer.schema import StrictModel
from seisbench_trainer.units import seconds_to_samples

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
    sampling_rate: PositiveFloat = Field(
        default=100.0,
        description="In Hz. Every dataset is resampled to it on access; the pick "
        "sample columns follow.",
    )
    resample_zerophase: bool = Field(
        default=True,
        description="Zero-phase anti-alias filter when a dataset is decimated "
        "(seisbench's default is causal, which delays the waveforms against the "
        "picks). True matches the default of the models' annotate, so training "
        "and inference see the same filter.",
    )
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


def _default_train_steps() -> list[TrainStep]:
    # pick-benchmark PhaseNet recipe: 2/3 pick context window, then the crop;
    # then half the windows rotated, real noise on half the pick windows (a
    # no-op for datasets without noise traces), and channels dropped from a
    # fifth of the windows (a missing component, after the noise: it stays
    # silent)
    return [
        PickWindow(probability=2 / 3),
        RandomWindow(),
        RotateHorizontal(probability=0.5),
        RealNoise(probability=0.5),
        ChannelDropout(probability=0.2),
    ]


class AugmentationConfig(StrictModel):
    """Training augmentations, normalization and labels. Durations in seconds."""

    train: list[TrainStep] = Field(
        default_factory=_default_train_steps,
        description="Training pipeline, applied in order: pick_window steps, "
        "exactly one random_window crop to the model input length, then the "
        "perturbations. Validation and test use fixed windows instead.",
    )
    window_s: PositiveFloat | None = Field(
        default=None,
        description="Model input length. None uses the model's own input length.",
    )
    normalization: Literal["peak", "std"] = Field(
        default="std",
        description="Per channel, after the training steps and in evaluation "
        "alike. PhaseNet then normalizes each channel again by its own "
        "``norm`` (annotate_batch_pre), as at inference, so for PhaseNet this "
        "does not change the model input.",
    )
    label_sigma_s: PositiveFloat = Field(
        default=0.1, description="Width of the pick label curve."
    )
    label_shape: LabelShape = "gaussian"
    missing_picks: Literal["unread", "absent"] = Field(
        default="unread",
        description="What a missing P or S pick means on a trace that has a "
        "pick of the other phase. 'unread' (as in the DLF v4b training): the "
        "analyst did not read it, but the arrival may be there. Where it can "
        "be, the training loss ignores all output channels and the validation "
        "and test metrics ignore that phase's peaks: before the first S pick "
        "if P is missing, after the first P pick if S is missing, each with a "
        "margin of 3 label_sigma_s. Otherwise the model would learn 'no P' "
        "before S on S-only traces (65 % of the Japan traces) and be scored "
        "down for picking it. 'absent': a missing pick means no arrival; the "
        "label is noise there and every peak of that phase is a false pick "
        "(seisbench pick-benchmark). Traces without any pick are noise in both.",
    )

    def window_samples(self, model_samples: int, sampling_rate: float) -> int:
        """Model input length in samples: ``window_s``, or the model's own."""
        if self.window_s is None:
            return model_samples
        return seconds_to_samples(self.window_s, sampling_rate)

    @model_validator(mode="after")
    def _windows_then_perturbations(self) -> Self:
        crops = [i for i, s in enumerate(self.train) if isinstance(s, RandomWindow)]
        if len(crops) != 1:
            raise ValueError("train needs exactly one random_window step")
        before, after = self.train[: crops[0]], self.train[crops[0] + 1 :]
        if not all(isinstance(step, PickWindow) for step in before):
            raise ValueError("only pick_window steps may come before random_window")
        if any(isinstance(step, PickWindow) for step in after):
            raise ValueError("pick_window steps must come before random_window")
        return self
