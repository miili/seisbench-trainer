"""Unit conversions. The only place where seconds become samples."""

from __future__ import annotations


def seconds_to_samples(seconds: float, sampling_rate: float) -> int:
    """Convert a duration in seconds to the nearest whole number of samples."""
    return round(seconds * sampling_rate)
