"""Training augmentation steps. Each step is one member of the ``TrainStep`` union.

``augmentation.train`` lists the steps in the order they are applied: any
``pick_window`` steps, one ``random_window`` crop to the model input, then the
perturbations. Every random step states its ``probability``: the chance that
it is applied to one training example; otherwise the example passes through
unchanged. Normalization and labels follow the list, so the steps see raw
waveforms. The windows are seisbench's; the perturbations are the reviewed
re-implementations in ``data.perturbations``.

seisbench is imported only when a window step is built, so that config
handling stays fast.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal, Self

import pandas as pd
from pydantic import (
    AfterValidator,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    model_validator,
)

from seisbench_trainer.data import perturbations
from seisbench_trainer.data.phases import PHASE_COLUMNS, Augmentation, StateDict
from seisbench_trainer.data.preprocessing import query_mask
from seisbench_trainer.schema import StrictModel
from seisbench_trainer.units import seconds_to_samples

if TYPE_CHECKING:
    from seisbench_trainer.data.loaders import AnyDataset

logger = logging.getLogger(__name__)

Probability = Annotated[float, Field(ge=0.0, le=1.0)]


def _ordered(value: tuple[float, float]) -> tuple[float, float]:
    if value[0] > value[1]:
        raise ValueError("expected (low, high) with low <= high")
    return value


Range = Annotated[tuple[NonNegativeFloat, NonNegativeFloat], AfterValidator(_ordered)]
"""A (low, high) range; the value is drawn uniformly from it per sample."""


@dataclass(frozen=True)
class StepContext:
    """Run settings a step may need to build its augmentation."""

    sampling_rate: float
    window_samples: int
    component_order: str
    train: AnyDataset
    """The training split (the noise source of ``real_noise``)."""


class RandomStep(StrictModel, ABC):
    """A step that is applied to a training example with ``probability``."""

    probability: Probability = Field(
        description="Chance that the step is applied to a training example."
    )

    def build(self, context: StepContext) -> Augmentation:
        return perturbations.Sometimes(self.build_step(context), self.probability)

    @abstractmethod
    def build_step(self, context: StepContext) -> Augmentation:
        """The augmentation, applied every time it is called."""


class PickWindow(RandomStep):
    """Cut a context window around a random pick (``WindowAroundSample``).

    Applied with ``probability``; otherwise the following ``random_window``
    crops from the whole trace, which may then contain no pick at all.
    """

    type: Literal["pick_window"] = "pick_window"
    length_s: PositiveFloat = Field(default=60.0, description="Window length.")
    before_s: NonNegativeFloat = Field(
        default=30.0, description="Window start before the selected pick."
    )

    @model_validator(mode="after")
    def _contains_pick(self) -> Self:
        if self.before_s > self.length_s:
            raise ValueError("before_s must not exceed length_s")
        return self

    def build_step(self, context: StepContext) -> Augmentation:
        import seisbench.generate as sbg

        rate = context.sampling_rate
        windowlen = seconds_to_samples(self.length_s, rate)
        if windowlen <= context.window_samples:
            # random_window could not move the pick: it would always sit
            # before_s into the model input
            raise ValueError(
                f"pick_window length_s ({self.length_s:g} s = {windowlen} samples) "
                f"must exceed the model input ({context.window_samples} samples "
                f"= {context.window_samples / rate:g} s at {rate:g} Hz); "
                "about twice the input is usual"
            )
        return sbg.WindowAroundSample(
            list(PHASE_COLUMNS),
            samples_before=seconds_to_samples(self.before_s, rate),
            windowlen=windowlen,
            selection="random",
            strategy="variable",
        )


class RandomWindow(StrictModel):
    """Crop a random window of the model input length; pad short traces.

    Always applied; exactly one per pipeline.
    """

    type: Literal["random_window"] = "random_window"

    def build(self, context: StepContext) -> Augmentation:
        import seisbench.generate as sbg

        return sbg.RandomWindow(windowlen=context.window_samples, strategy="pad")


class ChannelDropout(RandomStep):
    """Zero a random subset of the channels: at least one, never all."""

    type: Literal["channel_dropout"] = "channel_dropout"

    def build_step(self, context: StepContext) -> Augmentation:
        return perturbations.ChannelDropout()


class AddGap(RandomStep):
    """Zero all channels over a stretch (a data gap); picks in it are dropped."""

    type: Literal["add_gap"] = "add_gap"
    length_s: Range = Field(
        default=(0.5, 5.0),
        description="Gap length range. EQTransformer used gaps of up to 5 s.",
    )

    def build_step(self, context: StepContext) -> Augmentation:
        low, high = (
            seconds_to_samples(t, context.sampling_rate) for t in self.length_s
        )
        return perturbations.AddGap(low, high)


class GaussianNoise(RandomStep):
    """Add white noise; its std is ``scale`` times the waveform peak."""

    type: Literal["gaussian_noise"] = "gaussian_noise"
    scale: Range = Field(
        default=(0.0, 0.15),
        description="Noise std relative to the absolute peak of the waveform.",
    )

    def build_step(self, context: StepContext) -> Augmentation:
        return perturbations.GaussianNoise(*self.scale)


class RotateHorizontal(RandomStep):
    """Rotate N and E by a random angle in [-pi, pi)."""

    type: Literal["rotate_horizontal"] = "rotate_horizontal"

    def build_step(self, context: StepContext) -> Augmentation:
        return perturbations.RotateHorizontal(context.component_order)


def _unchanged(state_dict: StateDict) -> None:
    """Leave the sample as it is."""


class RealNoise(RandomStep):
    """Add a random window of a recorded noise trace from the training split.

    The noise's horizontal components are rotated by a random angle first.
    Windows without a pick (noise traces, pick-free crops) get no noise, so
    ``probability`` is the chance per window with a pick. If ``query``
    selects no training traces (or names a missing column), the step logs a
    warning and adds nothing.
    """

    type: Literal["real_noise"] = "real_noise"
    query: str = Field(
        default='trace_category == "noise"',
        description="Pandas query on the training split metadata that selects "
        "the noise traces. Dev and test traces are never used.",
    )
    scale: Range = Field(
        default=(0.0, 1.0),
        description="Noise peak relative to the absolute peak of the waveform.",
    )

    def build_step(self, context: StepContext) -> Augmentation:
        try:
            mask = query_mask(context.train.metadata, self.query)
        except pd.errors.UndefinedVariableError:  # e.g. no trace_category
            mask = None
        if mask is None or not mask.any():
            # real_noise is a default step: datasets without noise traces
            # (BohemiaSaxony, ETHZ) train without it instead of failing
            logger.warning(
                "real_noise query %r selects no training traces; no real noise "
                "is added",
                self.query,
            )
            return _unchanged
        noise = context.train.filter(mask, inplace=False)
        assert noise is not None
        return perturbations.RealNoise(
            noise, context.window_samples, *self.scale, context.component_order
        )


TrainStep = Annotated[
    PickWindow
    | RandomWindow
    | ChannelDropout
    | AddGap
    | GaussianNoise
    | RotateHorizontal
    | RealNoise,
    Field(discriminator="type"),
]
