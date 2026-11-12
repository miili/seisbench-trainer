"""Replace the dataset's train/dev/test split with a seeded random one."""

import logging
import math
from typing import TYPE_CHECKING, Annotated, Literal, Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import Field, NonNegativeInt, model_validator

from seisbench_trainer.data.preprocessing.base import (
    SPLITS,
    DatasetStep,
    Split,
)

if TYPE_CHECKING:
    import seisbench.data as sbd

logger = logging.getLogger(__name__)

Fraction = Annotated[float, Field(ge=0.0, le=1.0)]


class Resplit(DatasetStep):
    """Overwrite the metadata 'split' column with a seeded random assignment.

    Traces that share a ``group_by`` value (by default the event,
    ``source_id``) always land in the same split, so that one event recorded
    at many stations does not appear in both training and test. Traces
    without a group value (noise) are assigned one by one. Whole groups are
    shuffled and assigned in turn, so the trace fractions hold up to the size
    of one group. The result depends only on the seed and the metadata.
    """

    type: Literal["resplit"] = "resplit"
    fractions: dict[Split, Fraction] = Field(
        description="Share of the traces per split, summing to 1. A missing "
        "split gets no traces.",
        examples=[{"train": 0.7, "dev": 0.1, "test": 0.2}],
    )
    seed: NonNegativeInt = Field(
        default=0,
        description="Fixes the split; independent of training.seed, so that "
        "runs with different training seeds share one test set.",
    )
    group_by: str | None = Field(
        default="source_id",
        description="Metadata column whose traces stay together. None "
        "assigns every trace on its own.",
    )

    @model_validator(mode="after")
    def _fractions_sum_to_one(self) -> Self:
        if not math.isclose(sum(self.fractions.values()), 1.0, abs_tol=1e-6):
            raise ValueError(f"fractions must sum to 1, got {self.fractions}")
        return self

    def _groups(self, metadata: pd.DataFrame) -> NDArray[np.intp]:
        """A group index per trace; ungrouped traces get one group each."""
        if self.group_by is None:
            return np.arange(len(metadata))
        if self.group_by not in metadata.columns:
            raise ValueError(
                f"resplit group_by column {self.group_by!r} is not in the "
                "metadata; set group_by to null to split trace by trace"
            )
        # strings sort the same on every platform; missing values give -1
        codes, uniques = pd.factorize(
            metadata[self.group_by].astype("string"), sort=True
        )
        ungrouped = codes < 0
        codes[ungrouped] = len(uniques) + np.arange(np.count_nonzero(ungrouped))
        return codes

    def apply(self, dataset: "sbd.WaveformDataset") -> None:
        metadata = dataset.metadata
        groups = self._groups(metadata)
        sizes = np.bincount(groups)
        order = np.random.default_rng(self.seed).permutation(len(sizes))
        # each group goes to the split that holds the centre of its stretch
        ends = np.cumsum(sizes[order])
        centres = (ends - sizes[order] / 2) / max(len(metadata), 1)
        bounds = np.cumsum([self.fractions.get(split, 0.0) for split in SPLITS])
        position = np.searchsorted(bounds, centres, side="right")
        group_split = np.empty(len(sizes), dtype=int)
        group_split[order] = np.minimum(position, len(SPLITS) - 1)
        metadata["split"] = np.asarray(SPLITS)[group_split[groups]]
        counts = metadata["split"].value_counts()
        logger.info(
            "resplit (seed %d, group_by %s): %s",
            self.seed,
            self.group_by,
            ", ".join(f"{split} {counts.get(split, 0)}" for split in SPLITS),
        )
