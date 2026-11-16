"""Phase names, pick metadata columns and augmentation types.

Kept free of heavy imports (seisbench, torch), so config modules can use it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

StateDict = dict[str, tuple[NDArray[Any], dict[str, Any] | None]]
Augmentation = Callable[[StateDict], None]

PHASES = ("P", "S")
"""Phase order of the ``pick_samples`` batch key."""

PHASE_COLUMNS: dict[str, str] = {
    "trace_p_arrival_sample": "P",
    "trace_pP_arrival_sample": "P",
    "trace_P_arrival_sample": "P",
    "trace_P1_arrival_sample": "P",
    "trace_Pg_arrival_sample": "P",
    "trace_Pn_arrival_sample": "P",
    "trace_PmP_arrival_sample": "P",
    "trace_pwP_arrival_sample": "P",
    "trace_pwPm_arrival_sample": "P",
    "trace_s_arrival_sample": "S",
    "trace_S_arrival_sample": "S",
    "trace_S1_arrival_sample": "S",
    "trace_Sg_arrival_sample": "S",
    "trace_SmS_arrival_sample": "S",
    "trace_Sn_arrival_sample": "S",
}
"""Metadata pick columns and the phase each one labels."""


def phase_columns(phase: str) -> list[str]:
    """The metadata pick columns of one phase."""
    return [column for column, p in PHASE_COLUMNS.items() if p == phase]


def phase_picks(metadata: dict[str, Any] | None, phase: str) -> list[float]:
    """The finite picks of one phase in ``metadata``, in samples."""
    picks: list[float] = []
    for column in phase_columns(phase):
        value = (metadata or {}).get(column)
        if isinstance(value, int | np.integer | float) and np.isfinite(value):
            picks.append(float(value))
    return picks
