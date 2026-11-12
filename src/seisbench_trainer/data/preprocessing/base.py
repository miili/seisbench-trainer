"""Shared parts of the preprocessing steps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from seisbench_trainer.schema import StrictModel

if TYPE_CHECKING:
    import seisbench.data as sbd

Split = Literal["train", "dev", "test"]
SPLITS: tuple[Split, ...] = get_args(Split)


def query_mask(metadata: pd.DataFrame, query: str) -> NDArray[np.bool_]:
    """Rows of ``metadata`` that match a pandas query."""
    mask = metadata.eval(query)
    if not (isinstance(mask, pd.Series) and mask.dtype == bool):
        raise ValueError(f"query {query!r} does not give a boolean mask")
    return mask.to_numpy(dtype=bool)


class DatasetStep(StrictModel, ABC):
    """A step that changes a loaded dataset before it is split."""

    @abstractmethod
    def apply(self, dataset: sbd.WaveformDataset) -> None:
        """Change ``dataset`` in place."""
