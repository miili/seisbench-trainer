"""Picker model configs (one module per model family)."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from seisbench_trainer.models.phasenet import PhaseNetConfig

ModelConfig = Annotated[PhaseNetConfig, Field(discriminator="type")]

__all__ = ["ModelConfig", "PhaseNetConfig"]
