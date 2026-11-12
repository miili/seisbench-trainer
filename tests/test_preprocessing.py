from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import seisbench.data as sbd
from conftest import N_TRACES

from seisbench_trainer.data import DataConfig, LocalDataset, SyntheticDataset
from seisbench_trainer.data.loaders import load_splits
from seisbench_trainer.data.preprocessing import Filter, Resplit

SAMPLING_RATE = 100.0
NOISE = "trace_category == 'noise'"
FRACTIONS = {"train": 0.6, "dev": 0.15, "test": 0.25}


def load(path: Path, *steps: Filter | Resplit) -> sbd.WaveformDataset:
    source = LocalDataset(path=path, preprocessing=list(steps))
    return source.load("ZNE", SAMPLING_RATE)


def splits_of(dataset: sbd.WaveformDataset) -> dict[str, str]:
    metadata = dataset.metadata
    return dict(zip(metadata["trace_name"], metadata["split"], strict=True))


def test_resplit_follows_fractions_and_seed(synthetic_path: Path) -> None:
    original = splits_of(load(synthetic_path))
    step = Resplit(fractions=FRACTIONS, group_by=None, seed=1)
    first = splits_of(load(synthetic_path, step))
    assert first == splits_of(load(synthetic_path, step))
    assert first != original
    other_seed = Resplit(fractions=FRACTIONS, group_by=None, seed=2)
    assert first != splits_of(load(synthetic_path, other_seed))
    counts = pd.Series(first).value_counts()
    for split, fraction in FRACTIONS.items():
        assert abs(counts[split] - fraction * N_TRACES) <= 1


def test_resplit_keeps_groups_together(synthetic_path: Path) -> None:
    step = Resplit(fractions=FRACTIONS, group_by="station_code")
    metadata = load(synthetic_path, step).metadata
    assert (metadata.groupby("station_code")["split"].nunique() == 1).all()
    assert metadata["split"].nunique() > 1


def test_resplit_leaves_out_missing_split(synthetic_path: Path) -> None:
    step = Resplit(fractions={"train": 0.8, "dev": 0.2}, group_by=None)
    train, dev, test = load(synthetic_path, step).train_dev_test()
    assert len(test) == 0
    assert len(train) + len(dev) == N_TRACES


def test_resplit_needs_group_column(synthetic_path: Path) -> None:
    with pytest.raises(ValueError, match="'source_id' is not in the metadata"):
        load(synthetic_path, Resplit(fractions=FRACTIONS))


def test_ungrouped_traces_get_a_group_each() -> None:
    metadata = pd.DataFrame({"source_id": ["b", None, "a", "b", np.nan, "a"]})
    groups = Resplit(fractions=FRACTIONS)._groups(metadata)
    assert groups.tolist() == [1, 2, 0, 1, 3, 0]


def test_filter_on_train_keeps_other_splits(synthetic_path: Path) -> None:
    full = load(synthetic_path).train_dev_test()
    train, dev, test = load(
        synthetic_path, Filter(query=NOISE, splits=["train"])
    ).train_dev_test()
    assert 0 < len(train) < len(full[0])
    assert (train.metadata["trace_category"] == "noise").all()
    assert (len(dev), len(test)) == (len(full[1]), len(full[2]))


def test_filter_defaults_to_every_split(synthetic_path: Path) -> None:
    metadata = load(synthetic_path, Filter(query=NOISE)).metadata
    assert (metadata["trace_category"] == "noise").all()
    assert set(metadata["split"]) == {"train", "dev", "test"}


def test_filter_acts_on_the_new_split(synthetic_path: Path) -> None:
    """Steps run in order: the filter sees the split the resplit wrote."""
    resplit = Resplit(fractions=FRACTIONS, group_by=None, seed=5)
    dataset = load(synthetic_path, resplit, Filter(query=NOISE, splits=["train"]))
    expected = splits_of(load(synthetic_path, resplit))
    assert splits_of(dataset).items() <= expected.items()
    kept_events = dataset.metadata.query("trace_category == 'earthquake'")
    assert set(kept_events["split"]) == {"dev", "test"}


def test_filter_on_splits_needs_split_column() -> None:
    class NoSplits:
        metadata = pd.DataFrame({"trace_category": ["noise"]})

    dataset: Any = NoSplits()
    with pytest.raises(ValueError, match="needs a 'split' column"):
        Filter(query=NOISE, splits=["train"]).apply(dataset)


def test_load_splits_applies_preprocessing(synthetic_path: Path) -> None:
    source = LocalDataset(
        path=synthetic_path,
        preprocessing=[Resplit(fractions={"train": 0.5, "test": 0.5}, group_by=None)],
    )
    splits = load_splits(DataConfig(datasets=[source]), preload=False)
    assert len(splits.dev) == 0
    assert abs(len(splits.train) - N_TRACES / 2) <= 1


def test_synthetic_cache_ignores_preprocessing() -> None:
    step = Filter(query=NOISE)
    assert SyntheticDataset(preprocessing=[step]).path() == SyntheticDataset().path()
