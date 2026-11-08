"""Shared Pydantic base model for all config classes."""

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Immutable config model that rejects unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)
