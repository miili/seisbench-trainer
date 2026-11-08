"""Dataset sources. Each source is one member of the ``DatasetConfig`` union."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from seisbench_trainer.schema import StrictModel


class SeisBenchDataset(StrictModel):
    """A dataset class from ``seisbench.data``, downloaded on first use."""

    type: Literal["seisbench"] = "seisbench"
    name: str = Field(description="Class name in seisbench.data, e.g. 'ETHZ'.")
    query: str | None = Field(
        default=None,
        description="Optional pandas query on the metadata to select traces, "
        'e.g. "source_magnitude > 1".',
    )


class LocalDataset(StrictModel):
    """A SeisBench-format dataset in a local directory."""

    type: Literal["local"] = "local"
    path: Path = Field(description="Directory with metadata.csv and waveforms.hdf5.")
    query: str | None = Field(
        default=None,
        description="Optional pandas query on the metadata to select traces.",
    )


DatasetConfig = Annotated[SeisBenchDataset | LocalDataset, Field(discriminator="type")]
