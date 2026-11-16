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


def own_picks(metadata: dict[str, Any] | None, phase: str) -> list[float]:
    """The finite picks of one phase in the trace's own pick columns."""
    picks: list[float] = []
    for column in phase_columns(phase):
        value = (metadata or {}).get(column)
        if isinstance(value, int | np.integer | float) and np.isfinite(value):
            picks.append(float(value))
    return picks


def arrival_keys(phase: str) -> tuple[str, str]:
    """Metadata keys of a phase's arrival positions and event numbers."""
    return f"arrivals_{phase}_sample", f"arrivals_{phase}_event"


def phase_arrivals(
    metadata: dict[str, Any] | None, phase: str
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """The finite arrivals of one phase (in samples) and their event numbers,
    as ``data.arrivals.Arrivals`` wrote them; empty if it did not run."""
    sample_key, event_key = arrival_keys(phase)
    samples = np.asarray((metadata or {}).get(sample_key, []), dtype=np.float64)
    events = np.asarray((metadata or {}).get(event_key, []), dtype=np.int64)
    finite = np.isfinite(samples)
    return samples[finite], events[finite]
