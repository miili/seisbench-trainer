from collections.abc import Callable
from typing import Any

from numpy.typing import DTypeLike, NDArray
from seisbench.data import MultiWaveformDataset, WaveformDataset
from torch.utils.data import Dataset

_StateDict = dict[str, tuple[NDArray[Any], dict[str, Any] | None]]
_Augmentation = Callable[[_StateDict], None]
_Key = str | tuple[str, str]

class GenericGenerator(Dataset[dict[str, NDArray[Any]]]):
    dataset: WaveformDataset | MultiWaveformDataset
    def __init__(self, dataset: WaveformDataset | MultiWaveformDataset) -> None: ...
    def add_augmentations(self, augmentations: list[_Augmentation]) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> dict[str, NDArray[Any]]: ...

class FixedWindow:
    def __init__(
        self,
        p0: int | None = None,
        windowlen: int | None = None,
        strategy: str = "fail",
        axis: int = -1,
        key: _Key = "X",
    ) -> None: ...
    def __call__(
        self,
        state_dict: _StateDict,
        p0: int | None = None,
        windowlen: int | None = None,
    ) -> None: ...

class WindowAroundSample(FixedWindow):
    def __init__(
        self,
        metadata_keys: str | list[str],
        samples_before: int = 0,
        selection: str = "first",
        windowlen: int | None = None,
        strategy: str = "fail",
        axis: int = -1,
        key: _Key = "X",
    ) -> None: ...

class RandomWindow(FixedWindow):
    def __init__(
        self,
        low: int | None = None,
        high: int | None = None,
        windowlen: int | None = None,
        strategy: str = "fail",
        axis: int = -1,
        key: _Key = "X",
    ) -> None: ...

class Normalize:
    def __init__(
        self,
        demean_axis: int | None = None,
        detrend_axis: int | None = None,
        amp_norm_axis: int | None = None,
        amp_norm_type: str = "peak",
        eps: float = 1e-10,
        key: _Key = "X",
    ) -> None: ...
    def __call__(self, state_dict: _StateDict) -> None: ...

class ChangeDtype:
    def __init__(self, dtype: DTypeLike, key: _Key = "X") -> None: ...
    def __call__(self, state_dict: _StateDict) -> None: ...

class OneOf:
    def __init__(
        self,
        augmentations: list[_Augmentation],
        probabilities: list[float] | None = None,
    ) -> None: ...
    def __call__(self, state_dict: _StateDict) -> None: ...

class NullAugmentation:
    def __call__(self, state_dict: _StateDict) -> None: ...

class ProbabilisticLabeller:
    def __init__(
        self,
        shape: str = "gaussian",
        sigma: int = 10,
        label_columns: list[str] | dict[str, str] | None = None,
        noise_column: bool = True,
        model_labels: str | list[str] | None = None,
        dim: int = 1,
        key: tuple[str, str] = ("X", "y"),
    ) -> None: ...
    def __call__(self, state_dict: _StateDict) -> None: ...
