"""The arrivals that a trace's labels and metrics use.

``Arrivals`` is the first step of every pipeline. It writes, per phase,
``arrivals_<phase>_sample`` (positions in samples) and
``arrivals_<phase>_event`` (0 for the trace's own event, 1, 2, ... for other
events) into the metadata. The window steps shift the ``_sample`` arrays like
any pick column. Labels, pick metrics, unread masks, gaps and real noise read
only these arrays (``phases.phase_arrivals``).

``label_events="trace_event"``: the arrivals are the picks of the trace's own
metadata row, as in seisbench. ``"all_events"``: they also include the picks
of every other event recorded at the same station with an arrival inside the
trace, looked up in ``StationPicks``. A metadata row holds only its own
event, so in a swarm the arrivals of the other events in a window would be
labelled noise, and their detections counted as false picks.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from seisbench_trainer.data.phases import (
    PHASE_COLUMNS,
    PHASES,
    StateDict,
    arrival_keys,
    own_picks,
)

LabelEvents = Literal["trace_event", "all_events"]

STATION_COLUMNS = ("station_network_code", "station_code", "station_location_code")
REQUIRED_COLUMNS = (
    "trace_name",
    "trace_start_time",
    "trace_sampling_rate_hz",
    "station_network_code",
    "station_code",
)


class StationPicks:
    """Every pick of a dataset in absolute time, by station.

    Built from the full metadata (before any query or filter: the arrivals
    of a filtered-out event are still in the waveforms). An event is a
    ``source_id``; without that column every row is its own event. Picks of
    the trace's own event, also from other rows of the station, are left out
    of ``other_events``.
    """

    def __init__(self, metadata: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in metadata.columns]
        if missing:
            raise ValueError(
                f"label_events 'all_events' needs metadata columns {missing}; "
                "set augmentation.label_events to 'trace_event' for this dataset"
            )
        names = metadata["trace_name"].astype(str)
        if names.duplicated().any():
            raise ValueError("label_events 'all_events' needs unique trace names")
        parts = [
            metadata[c].astype("string").fillna("")
            for c in STATION_COLUMNS
            if c in metadata.columns
        ]
        station_key = parts[0].str.cat(parts[1:], sep=".")
        station = pd.factorize(station_key)[0]
        if "source_id" in metadata.columns:
            event = pd.factorize(metadata["source_id"].astype("string"))[0]
            no_event = event < 0  # rows without a source_id: one event each
            event[no_event] = event.max() + 1 + np.arange(np.count_nonzero(no_event))
        else:
            event = np.arange(len(metadata))
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        start = (
            (pd.to_datetime(metadata["trace_start_time"], utc=True) - epoch)
            / pd.Timedelta("1s")
        ).to_numpy(float)
        rate = metadata["trace_sampling_rate_hz"].to_numpy(float)

        stations, times, phases, events = [], [], [], []
        for column, phase in PHASE_COLUMNS.items():
            if column not in metadata.columns:
                continue
            sample = metadata[column].to_numpy(float)
            ok = np.isfinite(sample)
            stations.append(station[ok])
            times.append(start[ok] + sample[ok] / rate[ok])
            phases.append(np.full(np.count_nonzero(ok), PHASES.index(phase)))
            events.append(event[ok])
        all_stations = np.concatenate([np.zeros(0, np.int64), *stations])
        all_times = np.concatenate([np.zeros(0), *times])
        all_phases = np.concatenate([np.zeros(0, np.int64), *phases])
        all_events = np.concatenate([np.zeros(0, np.int64), *events])
        order = np.lexsort((all_times, all_stations))
        self.times: NDArray[np.float64] = all_times[order]
        self.phases: NDArray[np.int64] = all_phases[order]
        self.events: NDArray[np.int64] = all_events[order]
        sorted_stations = all_stations[order]
        n_stations = int(station.max()) + 1 if len(station) else 0
        self.bounds = np.searchsorted(sorted_stations, np.arange(n_stations + 1))
        # the longest time between two picks of one event at one station: the
        # picks of an event inside a trace lie within this of the trace
        by_event = pd.Series(all_times).groupby([all_stations, all_events])
        spans = by_event.max() - by_event.min()
        self.pad = float(spans.max()) if len(spans) else 0.0
        self.traces = dict(
            zip(names, zip(station, start, event, strict=True), strict=True)
        )

    def other_events(
        self, trace_name: str, n_samples: int, sampling_rate: float
    ) -> dict[str, tuple[NDArray[np.float64], NDArray[np.int64]]]:
        """Per phase, the picks (in samples of the trace) and event numbers
        (1, 2, ...) of the other events with an arrival inside the trace."""
        station, start, own = self.traces[trace_name]
        lo, hi = self.bounds[station], self.bounds[station + 1]
        end = start + n_samples / sampling_rate
        first, last = lo + np.searchsorted(
            self.times[lo:hi], [start - self.pad, end + self.pad]
        )
        times = self.times[first:last]
        events = self.events[first:last]
        phases = self.phases[first:last]
        inside = (times >= start) & (times < end) & (events != own)
        others = np.unique(events[inside])
        keep = np.isin(events, others)
        numbers = np.searchsorted(others, events[keep]) + 1
        samples = (times[keep] - start) * sampling_rate
        return {
            phase: (samples[phases[keep] == i], numbers[phases[keep] == i])
            for i, phase in enumerate(PHASES)
        }


class Arrivals:
    """Write the arrival arrays of a trace into its metadata.

    The trace's own picks (its pick columns) are event 0. With
    ``station_picks``, the picks of other events at the station inside the
    trace follow. Must run before the window steps.
    """

    def __init__(self, station_picks: StationPicks | None = None) -> None:
        self.station_picks = station_picks

    def __call__(self, state_dict: StateDict) -> None:
        x, metadata = state_dict["X"]
        metadata = dict(metadata or {})
        others = (
            None
            if self.station_picks is None
            else self.station_picks.other_events(
                str(metadata["trace_name"]),
                x.shape[-1],
                float(metadata["trace_sampling_rate_hz"]),
            )
        )
        for phase in PHASES:
            samples = np.asarray(own_picks(metadata, phase), dtype=np.float64)
            events = np.zeros(samples.size, dtype=np.int64)
            if others is not None:
                samples = np.concatenate([samples, others[phase][0]])
                events = np.concatenate([events, others[phase][1]])
            sample_key, event_key = arrival_keys(phase)
            metadata[sample_key] = samples
            metadata[event_key] = events
        state_dict["X"] = (x, metadata)
