"""PhaseNet (Zhu & Beroza, 2019)."""

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from seisbench_trainer.data.phases import PHASE_COLUMNS, Augmentation
from seisbench_trainer.models.base import LabelShape, PickerModelConfig

if TYPE_CHECKING:
    import seisbench.models as sbm
    from torch import Tensor

    from seisbench_trainer.data.loaders import Batch


class PhaseNetConfig(PickerModelConfig):
    type: Literal["PhaseNet"] = "PhaseNet"
    pretrained: str | None = Field(
        default=None,
        description="SeisBench pretrained weights to start from, e.g. 'original'. "
        "None trains from scratch.",
    )

    def variant(self) -> str:
        return "scratch" if self.pretrained is None else f"from-{self.pretrained}"

    def build(self, component_order: str, sampling_rate: float) -> "sbm.PhaseNet":
        import seisbench.models as sbm

        if self.pretrained is None:
            return sbm.PhaseNet(
                component_order=component_order, sampling_rate=sampling_rate
            )

        pretrained = sbm.PhaseNet.from_pretrained(self.pretrained)
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

    def label_augmentations(
        self, model: "sbm.WaveformModel", sigma_samples: int, shape: LabelShape
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

    def forward(self, model: "sbm.WaveformModel", x: "Tensor") -> "Tensor":
        prediction: Tensor = model(model.annotate_batch_pre(x, {}))
        return prediction

    def loss(self, pred: "Tensor", batch: "Batch", eps: float = 1e-5) -> "Tensor":
        """Cross-entropy between target and predicted curves (reference loss)."""
        import torch

        cross_entropy = batch["y"] * torch.log(pred + eps)
        return -cross_entropy.mean(dim=-1).sum(dim=-1).mean()
