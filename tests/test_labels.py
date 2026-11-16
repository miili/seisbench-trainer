from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import seisbench.generate as sbg

from seisbench_trainer.data.arrivals import Arrivals
from seisbench_trainer.data.labels import LabelShape, PickLabeller
from seisbench_trainer.data.phases import PHASE_COLUMNS


def random_metadata(rng: np.random.Generator, length: int) -> dict[str, Any]:
    """Picks in random columns: inside and outside the window, NaN, ints."""
    metadata: dict[str, Any] = {"trace_name": "x"}
    for column in rng.choice(list(PHASE_COLUMNS), size=rng.integers(0, 5)):
        kind = rng.integers(0, 4)
        if kind == 0:
            metadata[column] = np.nan
        elif kind == 1:
            metadata[column] = np.int64(rng.integers(0, length))
        else:
            metadata[column] = float(rng.uniform(-0.2 * length, 1.2 * length))
    return metadata


@pytest.mark.parametrize("shape", ["gaussian", "triangle", "box"])
@pytest.mark.parametrize("labels", ["PSN", "NPS", "SPN"])
def test_same_targets_as_seisbench(shape: LabelShape, labels: str) -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        length = int(rng.integers(50, 400))
        sigma = int(rng.integers(1, 30))
        x = np.zeros((3, length), dtype=np.float32)
        metadata = random_metadata(rng, length)
        ours: dict[str, Any] = {"X": (x, dict(metadata))}
        reference: dict[str, Any] = {"X": (x, dict(metadata))}
        Arrivals()(ours)
        PickLabeller(labels, sigma, shape)(ours)
        sbg.ProbabilisticLabeller(
            label_columns=PHASE_COLUMNS,
            sigma=sigma,
            shape=shape,
            dim=0,
            model_labels=labels,
        )(reference)
        assert ours["y"][0].dtype == reference["y"][0].dtype
        np.testing.assert_array_equal(ours["y"][0], reference["y"][0])


def test_every_pick_of_a_phase_is_labelled() -> None:
    x = np.zeros((3, 100), dtype=np.float32)
    state: dict[str, Any] = {
        "X": (x, {"trace_Pg_arrival_sample": 20.0, "trace_Pn_arrival_sample": 70.0})
    }
    Arrivals()(state)
    PickLabeller("PSN", 2, "gaussian")(state)
    y = state["y"][0]
    assert y[0, 20] == y[0, 70] == 1.0
    np.testing.assert_allclose(y.sum(axis=0), 1.0)
