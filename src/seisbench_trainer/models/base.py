"""Abstract base of all picker model configs.

A model config is both the config section and the adapter: it builds the
SeisBench model and defines everything that differs between model families.
The rest of the package uses only this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from seisbench_trainer.data.phases import PHASES, Augmentation
from seisbench_trainer.models.loss import LossFunction
from seisbench_trainer.schema import StrictModel

if TYPE_CHECKING:
    import seisbench.models as sbm
    from torch import Tensor

LabelShape = Literal["gaussian", "triangle", "box"]


class PickerModelConfig(StrictModel, ABC):
    pretrained: str | None = Field(
        default=None,
        description="Weights to start from: a SeisBench pretrained name (e.g. "
        "'original') or a local SeisBench model path without suffix (the "
        ".json/.pt pair). None trains from scratch.",
    )

    def weights_path(self) -> Path | None:
        """``pretrained`` as a local model path, if it is one."""
        if self.pretrained is None:
            return None
        path = Path(self.pretrained)
        return path if path.with_suffix(".json").is_file() else None

    @abstractmethod
    def build(self, component_order: str, sampling_rate: float) -> sbm.WaveformModel:
        """Build the model for data with this component order and sampling rate.

        The model's own metadata must state this component order, so that the
        exported model is fed the same order at inference.
        """

    @abstractmethod
    def load_pretrained(self) -> sbm.WaveformModel:
        """The pretrained model in its own component order (e.g. a baseline).

        Raises ``ValueError`` if no pretrained weights are configured.
        """

    @abstractmethod
    def label_augmentations(
        self, model: sbm.WaveformModel, sigma_samples: int, shape: LabelShape
    ) -> list[Augmentation]:
        """Augmentations that write the training target ``y``."""

    @abstractmethod
    def forward_logits(self, model: sbm.WaveformModel, x: Tensor) -> Tensor:
        """Raw model output before the final activation; the loss input."""

    @abstractmethod
    def activate(self, logits: Tensor) -> Tensor:
        """Probabilities from ``forward_logits``, shape (batch, channel, sample)."""

    def forward(self, model: sbm.WaveformModel, x: Tensor) -> Tensor:
        """Model output as probabilities, shape (batch, channel, sample)."""
        return self.activate(self.forward_logits(model, x))

    @abstractmethod
    def build_loss(
        self, model: sbm.WaveformModel, sampling_rate: float
    ) -> LossFunction:
        """The configured training loss on ``forward_logits`` and the batch."""

    def phase_probabilities(
        self, model: sbm.WaveformModel, pred: Tensor
    ) -> dict[str, Tensor]:
        """Per-phase probability curves (batch, sample), looked up by label."""
        return {phase: pred[:, model.labels.index(phase)] for phase in PHASES}

    def weights_label(self) -> str:
        """Name of the starting weights: 'scratch', 'original', or a file name."""
        if self.pretrained is None:
            return "scratch"
        path = self.weights_path()
        return path.name if path else self.pretrained

    def variant(self) -> str:
        """Short description of the starting weights, e.g. 'from-original'."""
        label = self.weights_label()
        return label if self.pretrained is None else f"from-{label}"

    def window_samples(self, model: sbm.WaveformModel) -> int:
        """Default input window length."""
        return model.in_samples
