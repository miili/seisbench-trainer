"""Phase names, pick metadata columns and augmentation types.

Kept free of heavy imports (seisbench, torch), so config modules can use it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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
