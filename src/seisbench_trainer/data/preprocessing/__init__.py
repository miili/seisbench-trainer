"""Dataset preprocessing. Each step is one member of the ``PreprocessingStep`` union.

Every dataset entry in ``data.datasets`` lists its own ``preprocessing`` steps.
They run in order on the loaded dataset, after its ``query`` and before it is
split with ``train_dev_test()``, and see the metadata as the previous step
left it: a ``filter`` after a ``resplit`` acts on the new splits.

seisbench is not imported here, so that config handling stays fast.
"""

from typing import Annotated

from pydantic import Field

from seisbench_trainer.data.preprocessing.base import SPLITS, Split, query_mask
from seisbench_trainer.data.preprocessing.filter import Filter
from seisbench_trainer.data.preprocessing.resplit import Resplit

PreprocessingStep = Annotated[Filter | Resplit, Field(discriminator="type")]

__all__ = [
    "SPLITS",
    "Filter",
    "PreprocessingStep",
    "Resplit",
    "Split",
    "query_mask",
]
