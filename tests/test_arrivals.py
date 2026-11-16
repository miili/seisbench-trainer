"""Arrivals of other events in a trace (augmentation.label_events)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import seisbench.data as sbd

from seisbench_trainer.data import DataConfig, LocalDataset
from seisbench_trainer.data.arrivals import Arrivals
from seisbench_trainer.data.augmentations import PickSamples, UnreadPhaseMask
from seisbench_trainer.data.loaders import load_splits
from seisbench_trainer.data.phases import PHASES, StateDict, phase_arrivals

RATE = 100.0
LENGTH = 12000  # 120 s

# event, station, trace start (s after 2020-01-01), P and S in trace samples
TRACES = [
    ("e1", "AAA", 0.0, 6000.0, 6400.0),
    ("e2", "AAA", 15.0, 6000.0, np.nan),  # P at 75 s, no S: inside e1's trace
    ("e3", "BBB", 0.0, 6500.0, 6900.0),  # another station at the same time
    ("e4", "AAA", 3600.0, 6000.0, 6300.0),  # an hour later
]


@pytest.fixture
def swarm(tmp_path: Path) -> Path:
    """Four event traces; two events overlap at station AAA."""
    path = tmp_path / "swarm"
    with sbd.WaveformDataWriter(path / "metadata.csv", path / "waveforms.hdf5") as w:
        w.data_format = {"dimension_order": "CW", "component_order": "ZNE"}
        for event, station, start, p, s in TRACES:
            begin = np.datetime64("2020-01-01T00:00:00") + np.timedelta64(
                int(start * 1e6), "us"
            )
            w.add_trace(
                {
                    "trace_name": f"{event}_{station}",
                    "source_id": event,
                    "split": "train",
                    "station_network_code": "XX",
                    "station_code": station,
                    "trace_start_time": f"{begin}Z",
                    "trace_sampling_rate_hz": RATE,
                    "trace_category": "earthquake",
                    "trace_P_arrival_sample": p,
                    "trace_S_arrival_sample": s,
                },
                np.zeros((3, LENGTH), dtype=np.float32),
            )
    return path


def arrivals(
    path: Path, rate: float = RATE, query: str | None = None
) -> dict[str, dict[str, tuple[list[float], list[int]]]]:
    """Arrivals per trace and phase, as the pipelines see them."""
    config = DataConfig(
        datasets=[LocalDataset(path=path, query=query)], sampling_rate=rate
    )
    splits = load_splits(config, preload=False, label_events="all_events")
    out = {}
    for index in range(len(splits.train)):
        state: StateDict = {"X": splits.train.get_sample(index)}
        Arrivals(splits.station_picks)(state)
        metadata = state["X"][1]
        assert metadata is not None
        out[str(metadata["trace_name_original"])] = {
            phase: tuple(a.tolist() for a in phase_arrivals(metadata, phase))
            for phase in PHASES
        }
    return out  # type: ignore[return-value]


def test_other_events_at_the_station_are_arrivals(swarm: Path) -> None:
    found = arrivals(swarm)
    assert found["e1_AAA"] == {"P": ([6000.0, 7500.0], [0, 1]), "S": ([6400.0], [0])}
    # e1 seen from e2's trace, which starts 15 s later
    assert found["e2_AAA"] == {
        "P": ([6000.0, 4500.0], [0, 1]),
        "S": ([4900.0], [1]),
    }
    # other station, and an hour later: only the own event
    assert found["e3_BBB"] == {"P": ([6500.0], [0]), "S": ([6900.0], [0])}
    assert found["e4_AAA"] == {"P": ([6000.0], [0]), "S": ([6300.0], [0])}


def test_arrivals_follow_the_sampling_rate(swarm: Path) -> None:
    found = arrivals(swarm, rate=50.0)
    assert found["e1_AAA"]["P"] == ([3000.0, 3750.0], [0, 1])


def test_events_left_out_by_the_query_are_still_arrivals(swarm: Path) -> None:
    found = arrivals(swarm, query='source_id != "e2"')
    assert "e2_AAA" not in found
    assert found["e1_AAA"]["P"] == ([6000.0, 7500.0], [0, 1])


def test_unread_mask_per_event() -> None:
    # event 0 complete; event 1 has P but no S: only its S stretch is unread
    metadata: dict[str, Any] = {
        "arrivals_P_sample": np.array([100.0, 600.0]),
        "arrivals_P_event": np.array([0, 1]),
        "arrivals_S_sample": np.array([300.0]),
        "arrivals_S_event": np.array([0]),
    }
    state: StateDict = {"X": (np.ones((3, 1000)), metadata)}
    UnreadPhaseMask(margin=30)(state)
    mask = state["label_mask"][0]
    assert mask[PHASES.index("P")].all()
    s_row = mask[PHASES.index("S")]
    assert s_row[:630].all() and not s_row[630:].any()


def test_pick_samples_keep_every_event() -> None:
    metadata: dict[str, Any] = {
        "arrivals_P_sample": np.array([600.0, 100.0, np.nan, 2000.0]),
        "arrivals_P_event": np.array([1, 0, 2, 3]),
    }
    state: StateDict = {"X": (np.ones((3, 1000)), metadata)}
    PickSamples(1000)(state)
    row = state["pick_samples"][0][PHASES.index("P")]
    assert row[:2].tolist() == [100.0, 600.0] and np.isnan(row[2:]).all()


def test_all_events_needs_start_times(synthetic_path: Path) -> None:
    config = DataConfig(datasets=[LocalDataset(path=synthetic_path)])
    with pytest.raises(ValueError, match="trace_start_time"):
        load_splits(config, preload=False, label_events="all_events")
