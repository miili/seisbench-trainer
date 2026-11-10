"""Dataset sources. Each source is one member of the ``DatasetConfig`` union.

Every source loads a ``seisbench`` dataset with the run's component order and
sampling rate and applies its optional metadata query.
"""

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import Field, NonNegativeInt, PositiveFloat, PositiveInt

from seisbench_trainer.schema import StrictModel

# seisbench imports torch, which takes seconds. It is imported only when a
# dataset is actually loaded, so that config handling stays fast.
if TYPE_CHECKING:
    import seisbench.data as sbd


def query_mask(metadata: pd.DataFrame, query: str) -> NDArray[np.bool_]:
    """Rows of ``metadata`` that match a pandas query."""
    mask = metadata.eval(query)
    if not (isinstance(mask, pd.Series) and mask.dtype == bool):
        raise ValueError(f"query {query!r} does not give a boolean mask")
    return mask.to_numpy(dtype=bool)


def _apply_query(
    dataset: "sbd.WaveformDataset", query: str | None
) -> "sbd.WaveformDataset":
    if query is not None:
        dataset.filter(query_mask(dataset.metadata, query), inplace=True)
    return dataset


class SeisBenchDataset(StrictModel):
    """A dataset class from ``seisbench.data``, downloaded on first use."""

    type: Literal["seisbench"] = "seisbench"
    name: str = Field(description="Class name in seisbench.data, e.g. 'ETHZ'.")
    query: str | None = Field(
        default=None,
        description="Optional pandas query on the metadata to select traces, "
        'e.g. "source_magnitude > 1".',
    )

    def label(self) -> str:
        """Short name for experiment and model names."""
        return self.name

    def load(self, component_order: str, sampling_rate: float) -> "sbd.WaveformDataset":
        import seisbench.data as sbd

        dataset_class = getattr(sbd, self.name, None)
        if not (
            isinstance(dataset_class, type)
            and issubclass(dataset_class, sbd.WaveformBenchmarkDataset)
        ):
            raise ValueError(f"{self.name!r} is not a dataset class in seisbench.data")
        dataset = dataset_class(
            component_order=component_order, sampling_rate=sampling_rate, cache="full"
        )
        return _apply_query(dataset, self.query)


class LocalDataset(StrictModel):
    """A SeisBench-format dataset in a local directory."""

    type: Literal["local"] = "local"
    path: Path = Field(description="Directory with metadata.csv and waveforms.hdf5.")
    query: str | None = Field(
        default=None,
        description="Optional pandas query on the metadata to select traces.",
    )

    def label(self) -> str:
        """Short name for experiment and model names."""
        return self.path.name

    def load(self, component_order: str, sampling_rate: float) -> "sbd.WaveformDataset":
        import seisbench.data as sbd

        dataset = sbd.WaveformDataset(
            self.path,
            component_order=component_order,
            sampling_rate=sampling_rate,
            cache="full",
        )
        return _apply_query(dataset, self.query)


class SyntheticDataset(StrictModel):
    """Generated test data (noise plus P and S wavelets); no download needed."""

    type: Literal["synthetic"] = "synthetic"
    n_traces: PositiveInt = 300
    trace_length_s: float = Field(default=60.0, ge=30.0)
    sampling_rate: PositiveFloat = 100.0
    noise_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    seed: NonNegativeInt = 0
    query: str | None = None

    def label(self) -> str:
        """Short name for experiment and model names."""
        return "synthetic"

    def path(self) -> Path:
        """Cache directory, unique for the generation parameters."""
        params = self.model_dump_json(exclude={"query"}).encode()
        digest = hashlib.sha256(params).hexdigest()[:12]
        return Path(tempfile.gettempdir()) / "seisbench-trainer" / f"synthetic-{digest}"

    def ensure_written(self) -> Path:
        """Write the dataset once; safe when several processes race."""
        from seisbench_trainer.data.synthetic import write_synthetic_dataset

        path = self.path()
        if not path.exists():
            tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
            write_synthetic_dataset(
                tmp,
                n_traces=self.n_traces,
                trace_length_s=self.trace_length_s,
                sampling_rate=self.sampling_rate,
                noise_fraction=self.noise_fraction,
                seed=self.seed,
            )
            try:
                tmp.rename(path)
            except OSError:  # another process won the race
                shutil.rmtree(tmp)
        return path

    def load(self, component_order: str, sampling_rate: float) -> "sbd.WaveformDataset":
        local = LocalDataset(path=self.ensure_written(), query=self.query)
        return local.load(component_order, sampling_rate)


DatasetConfig = Annotated[
    SeisBenchDataset | LocalDataset | SyntheticDataset, Field(discriminator="type")
]
