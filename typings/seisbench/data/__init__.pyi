from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray

class WaveformDataset:
    def __init__(
        self,
        path: Path | str | None = None,
        name: str | None = None,
        dimension_order: str | None = None,
        component_order: str | None = None,
        sampling_rate: float | None = None,
        cache: Literal["full", "trace"] | None = None,
        chunks: list[str] | None = None,
        missing_components: Literal["pad", "copy", "ignore"] = "pad",
        metadata_cache: bool = False,
        resample_zerophase: bool = False,
        **kwargs: Any,
    ) -> None: ...
    @property
    def metadata(self) -> pd.DataFrame: ...
    def __len__(self) -> int: ...
    def __add__(
        self, other: WaveformDataset | MultiWaveformDataset
    ) -> MultiWaveformDataset: ...
    def filter(self, mask: NDArray[np.bool_], inplace: bool = True) -> Self | None: ...
    def train_dev_test(self) -> tuple[Self, Self, Self]: ...
    def preload_waveforms(self, pbar: bool = False) -> None: ...
    def get_sample(
        self, idx: int, sampling_rate: float | None = None
    ) -> tuple[NDArray[Any], dict[str, Any]]: ...

class WaveformBenchmarkDataset(WaveformDataset):
    def __init__(
        self,
        chunks: list[str] | None = None,
        citation: str | None = None,
        license: str | None = None,
        force: bool = False,
        wait_for_file: bool = False,
        repository_lookup: bool = False,
        compile_from_source: bool = False,
        download_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None: ...

class MultiWaveformDataset:
    def __init__(self, datasets: list[WaveformDataset]) -> None: ...
    @property
    def metadata(self) -> pd.DataFrame: ...
    def __len__(self) -> int: ...
    def __add__(
        self, other: WaveformDataset | MultiWaveformDataset
    ) -> MultiWaveformDataset: ...
    def filter(self, mask: NDArray[np.bool_], inplace: bool = True) -> Self | None: ...
    def train_dev_test(self) -> tuple[Self, Self, Self]: ...
    def preload_waveforms(self, pbar: bool = False) -> None: ...
    def get_sample(
        self, idx: int, sampling_rate: float | None = None
    ) -> tuple[NDArray[Any], dict[str, Any]]: ...

class WaveformDataWriter:
    data_format: dict[str, str]
    def __init__(
        self, metadata_path: Path | str, waveforms_path: Path | str
    ) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *args: object) -> None: ...
    def add_trace(self, metadata: dict[str, Any], waveform: NDArray[Any]) -> None: ...
