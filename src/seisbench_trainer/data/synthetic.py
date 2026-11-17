"""Synthetic SeisBench dataset for tests and laptop smoke runs.

Each event trace has Gaussian noise plus a P and an S wavelet. P is strongest
on Z, S on the horizontals. A fraction of the traces is pure noise (no picks).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import seisbench.data as sbd
from numpy.typing import NDArray

COMPONENT_ORDER = "ZNE"
START = np.datetime64("2020-01-01T00:00:00", "ms")
SPLIT_PROBABILITIES = {"train": 0.7, "dev": 0.15, "test": 0.15}


def _wavelet(
    n_samples: int, sampling_rate: float, frequency: float, decay_s: float
) -> NDArray[np.float64]:
    t = np.arange(n_samples) / sampling_rate
    return np.exp(-t / decay_s) * np.sin(2 * np.pi * frequency * t)


def _add_wavelet(
    trace: NDArray[np.float64],
    onset: int,
    wavelet: NDArray[np.float64],
    weights: tuple[float, float, float],
) -> None:
    end = min(trace.shape[1], onset + wavelet.size)
    for channel, weight in enumerate(weights):
        trace[channel, onset:end] += weight * wavelet[: end - onset]


def write_synthetic_dataset(
    path: Path,
    n_traces: int = 300,
    trace_length_s: float = 60.0,
    sampling_rate: float = 100.0,
    noise_fraction: float = 0.1,
    seed: int = 0,
) -> None:
    """Write a synthetic dataset (metadata.csv, waveforms.hdf5) to ``path``."""
    rng = np.random.default_rng(seed)
    n_samples = round(trace_length_s * sampling_rate)
    wavelet = _wavelet(round(3.0 * sampling_rate), sampling_rate, 5.0, 0.4)
    splits = list(SPLIT_PROBABILITIES)
    gap = np.timedelta64(round(2000 * trace_length_s), "ms")
    probabilities = list(SPLIT_PROBABILITIES.values())

    with sbd.WaveformDataWriter(
        path / "metadata.csv", path / "waveforms.hdf5"
    ) as writer:
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": COMPONENT_ORDER,
            "measurement": "velocity",
            "unit": "counts",
            "instrument_response": "not restituted",
        }
        for index in range(n_traces):
            trace = rng.normal(0.0, 1.0, size=(3, n_samples))
            metadata: dict[str, str | float] = {
                "trace_name": f"synthetic_{index:05d}",
                "split": str(rng.choice(splits, p=probabilities)),
                "station_network_code": "SY",
                "station_code": f"S{index % 10:02d}",
                # one trace after the other: no trace overlaps another
                "trace_start_time": f"{START + gap * index}Z",
                "trace_sampling_rate_hz": sampling_rate,
            }
            if rng.random() >= noise_fraction:
                p_sample = int(rng.uniform(5.0, trace_length_s - 20.0) * sampling_rate)
                s_sample = p_sample + int(rng.uniform(1.5, 8.0) * sampling_rate)
                amplitude = 10 ** rng.uniform(0.5, 1.5)
                _add_wavelet(trace, p_sample, amplitude * wavelet, (1.0, 0.3, 0.3))
                _add_wavelet(trace, s_sample, 2 * amplitude * wavelet, (0.3, 1.0, 1.0))
                metadata |= {
                    "trace_category": "earthquake",
                    "trace_p_arrival_sample": p_sample,
                    "trace_s_arrival_sample": s_sample,
                    "source_magnitude": rng.uniform(0.0, 3.0),
                }
            else:
                metadata["trace_category"] = "noise"
            writer.add_trace(metadata, trace.astype(np.float32))
