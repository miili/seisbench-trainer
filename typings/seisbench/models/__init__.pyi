from pathlib import Path
from typing import Any, Self

from torch import Tensor, nn

class SeisBenchModel(nn.Module):
    @classmethod
    def from_pretrained(
        cls,
        name: str,
        version_str: str = "latest",
        update: bool = False,
        force: bool = False,
        wait_for_file: bool = False,
    ) -> Self: ...
    @classmethod
    def load(
        cls, path: Path | str, version_str: str | None = None, **kwargs: Any
    ) -> Self: ...
    def save(
        self,
        path: Path | str,
        weights_docstring: str = "",
        version_str: str | None = None,
    ) -> None: ...
    def get_model_args(self) -> dict[str, Any]: ...

class WaveformModel(SeisBenchModel):
    sampling_rate: float
    in_samples: int
    labels: str
    @property
    def component_order(self) -> str: ...
    def annotate_batch_pre(self, batch: Tensor, argdict: dict[str, Any]) -> Tensor: ...

class PhaseNet(WaveformModel):
    def __init__(
        self,
        in_channels: int = 3,
        classes: int = 3,
        phases: str = "NPS",
        sampling_rate: float = 100,
        norm: str = "std",
        filter_factor: int = 1,
        **kwargs: Any,
    ) -> None: ...
    def forward(self, x: Tensor, logits: bool = False) -> Tensor: ...
