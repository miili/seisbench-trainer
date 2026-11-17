"""Shared Pydantic base model for all config classes."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Immutable config model that rejects unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
"""A probability or share, from 0 to 1."""

UserPath = Annotated[Path, AfterValidator(Path.expanduser)]
"""A path in which ``~`` stands for the home directory."""
