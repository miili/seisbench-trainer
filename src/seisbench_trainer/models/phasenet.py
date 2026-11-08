"""PhaseNet (Zhu & Beroza, 2019)."""

from typing import Literal

from pydantic import Field

from seisbench_trainer.schema import StrictModel


class PhaseNetConfig(StrictModel):
    type: Literal["PhaseNet"] = "PhaseNet"
    pretrained: str | None = Field(
        default=None,
        description="SeisBench pretrained weights to start from, e.g. 'original'. "
        "None trains from scratch.",
    )
