"""Keep only the traces that match a metadata query."""

import logging
from typing import TYPE_CHECKING, Literal, Self

from pydantic import Field, model_validator

from seisbench_trainer.data.preprocessing.base import (
    SPLITS,
    DatasetStep,
    Split,
    query_mask,
)

if TYPE_CHECKING:
    import seisbench.data as sbd

logger = logging.getLogger(__name__)


class Filter(DatasetStep):
    """Keep the traces of ``splits`` that match ``query``; other splits stay whole.

    For example, ``trace_category == "noise"`` on ``["train"]`` trains on
    noise only while validation and test keep every trace.
    """

    type: Literal["filter"] = "filter"
    query: str = Field(
        description="Pandas query on the metadata; matching traces are kept, "
        "e.g. 'trace_category == \"noise\"'."
    )
    splits: list[Split] = Field(
        default_factory=lambda: list(SPLITS),
        min_length=1,
        description="Splits the filter acts on, by the metadata 'split' column "
        "as it is at this step.",
    )

    @model_validator(mode="after")
    def _unique_splits(self) -> Self:
        if len(set(self.splits)) != len(self.splits):
            raise ValueError(f"splits must be unique, got {self.splits}")
        return self

    def apply(self, dataset: "sbd.WaveformDataset") -> None:
        metadata = dataset.metadata
        keep = query_mask(metadata, self.query)
        if set(self.splits) != set(SPLITS):
            if "split" not in metadata.columns:
                raise ValueError(
                    f"filter on splits {self.splits} needs a 'split' column; "
                    "add a resplit step before it"
                )
            keep = keep | ~metadata["split"].isin(self.splits).to_numpy()
        n_before = len(dataset)
        dataset.filter(keep, inplace=True)
        logger.info(
            "filter %r on %s: kept %d of %d traces",
            self.query,
            "/".join(self.splits),
            len(dataset),
            n_before,
        )
