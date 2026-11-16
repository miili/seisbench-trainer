"""PhaseNet (Zhu & Beroza, 2019)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from seisbench_trainer.data.labels import LabelShape, PickLabeller
from seisbench_trainer.data.phases import Augmentation
from seisbench_trainer.models.base import PickerModelConfig
from seisbench_trainer.models.loss import CrossEntropyLoss, LossFunction, PhaseNetLoss

if TYPE_CHECKING:
    import seisbench.models as sbm
    from torch import Tensor


class PhaseNetConfig(PickerModelConfig):
    type: Literal["PhaseNet"] = "PhaseNet"
    loss: PhaseNetLoss = Field(default_factory=CrossEntropyLoss)

    def build(self, component_order: str, sampling_rate: float) -> sbm.PhaseNet:
        import seisbench.models as sbm

        if self.pretrained is None:
            return sbm.PhaseNet(
                component_order=component_order, sampling_rate=sampling_rate
            )

        pretrained = self.load_pretrained()
        if pretrained.sampling_rate != sampling_rate:
            raise ValueError(
                f"pretrained {self.pretrained!r} expects "
                f"{pretrained.sampling_rate} Hz, the data has {sampling_rate} Hz; "
                f'set "model.sampling_rate" to {sampling_rate:g} to run the '
                "weights time-stretched"
            )
        if pretrained.component_order != component_order:
            # Feeding another order would swap the channels the weights expect.
            raise ValueError(
                f"pretrained {self.pretrained!r} expects component order "
                f"{pretrained.component_order!r}, the data has {component_order!r}; "
                f'set "data.component_order" to {pretrained.component_order!r}'
            )
        return pretrained

    def load_weights(self) -> sbm.PhaseNet:
        import seisbench.models as sbm

        assert self.pretrained is not None
        path = self.weights_path()
        if path is not None:
            return sbm.PhaseNet.load(path)
        return sbm.PhaseNet.from_pretrained(self.pretrained)

    def label_augmentations(
        self, model: sbm.WaveformModel, sigma_samples: int, shape: LabelShape
    ) -> list[Augmentation]:
        return [PickLabeller(model.labels, sigma_samples, shape)]

    def forward_logits(self, model: sbm.WaveformModel, x: Tensor) -> Tensor:
        logits: Tensor = model(model.annotate_batch_pre(x, {}), logits=True)
        return logits

    def activate(self, logits: Tensor) -> Tensor:
        return logits.softmax(dim=1)

    def build_loss(self, model: sbm.WaveformModel) -> LossFunction:
        return self.loss.build(model.labels)
