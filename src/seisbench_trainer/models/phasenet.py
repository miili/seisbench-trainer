"""PhaseNet (Zhu & Beroza, 2019)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from seisbench_trainer.data.phases import PHASE_COLUMNS, Augmentation
from seisbench_trainer.models.base import LabelShape, PickerModelConfig
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
                f"{pretrained.sampling_rate} Hz, the data has {sampling_rate} Hz"
            )
        if pretrained.component_order != component_order:
            # Feeding another order would swap the channels the weights expect.
            raise ValueError(
                f"pretrained {self.pretrained!r} expects component order "
                f"{pretrained.component_order!r}, the data has {component_order!r}; "
                f'set "data.component_order" to {pretrained.component_order!r}'
            )
        return pretrained

    def load_pretrained(self) -> sbm.PhaseNet:
        import seisbench.models as sbm

        if self.pretrained is None:
            raise ValueError("no pretrained weights configured")
        path = self.weights_path()
        if path is not None:
            return sbm.PhaseNet.load(path)
        return sbm.PhaseNet.from_pretrained(self.pretrained)

    def label_augmentations(
        self, model: sbm.WaveformModel, sigma_samples: int, shape: LabelShape
    ) -> list[Augmentation]:
        import seisbench.generate as sbg

        return [
            sbg.ProbabilisticLabeller(
                label_columns=PHASE_COLUMNS,
                sigma=sigma_samples,
                shape=shape,
                dim=0,
                model_labels=model.labels,
            )
        ]

    def forward_logits(self, model: sbm.WaveformModel, x: Tensor) -> Tensor:
        logits: Tensor = model(model.annotate_batch_pre(x, {}), logits=True)
        return logits

    def activate(self, logits: Tensor) -> Tensor:
        return logits.softmax(dim=1)

    def build_loss(
        self, model: sbm.WaveformModel, sampling_rate: float
    ) -> LossFunction:
        return self.loss.build(model.labels, sampling_rate)
