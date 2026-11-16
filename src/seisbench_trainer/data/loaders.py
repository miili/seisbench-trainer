"""Load, split, concatenate and preload datasets; build DataLoaders."""

from __future__ import annotations

import logging
import operator
from functools import reduce
from typing import NamedTuple, cast

import pandas as pd
import seisbench.data as sbd
import seisbench.generate as sbg
import torch
from seisbench.util import worker_seeding
from torch.utils.data import DataLoader

from seisbench_trainer.data.arrivals import LabelEvents, StationPicks
from seisbench_trainer.data.config import DataConfig
from seisbench_trainer.data.phases import Augmentation

logger = logging.getLogger(__name__)

AnyDataset = sbd.WaveformDataset | sbd.MultiWaveformDataset
Batch = dict[str, torch.Tensor]


class Splits(NamedTuple):
    train: AnyDataset
    dev: AnyDataset
    test: AnyDataset
    station_picks: StationPicks | None = None
    """Picks of every event by station (``label_events="all_events"``)."""


def _station_picks(config: DataConfig) -> StationPicks:
    """Picks of all datasets, before their queries and preprocessing: the
    arrivals of an event left out of the splits are still in the waveforms."""
    metadata = [
        source.model_copy(update={"query": None, "preprocessing": []})
        .load(config.component_order, config.sampling_rate, config.resample_zerophase)
        .metadata
        for source in config.datasets
    ]
    return StationPicks(pd.concat(metadata, ignore_index=True))


def load_splits(
    config: DataConfig,
    preload: bool = True,
    label_events: LabelEvents = "trace_event",
) -> Splits:
    """Load every dataset, split each one, and concatenate the splits by type.

    With ``label_events="all_events"`` the splits carry the ``StationPicks``
    that the pipelines need to find the other events in a trace.
    """
    parts: list[tuple[AnyDataset, AnyDataset, AnyDataset]] = []
    for source in config.datasets:
        dataset = source.load(
            config.component_order, config.sampling_rate, config.resample_zerophase
        )
        train, dev, test = dataset.train_dev_test()
        parts.append((train, dev, test))
    train, dev, test = (
        reduce(operator.add, split) for split in zip(*parts, strict=True)
    )
    station_picks = _station_picks(config) if label_events == "all_events" else None
    for name, split in (("train", train), ("dev", dev), ("test", test)):
        logger.info("%s split: %d traces", name, len(split))
        if preload:
            split.preload_waveforms(pbar=True)
    return Splits(train, dev, test, station_picks)


def make_loader(
    dataset: AnyDataset,
    augmentations: list[Augmentation],
    config: DataConfig,
    shuffle: bool,
) -> DataLoader[Batch]:
    """DataLoader over a dataset with the given augmentation pipeline."""
    generator = sbg.GenericGenerator(dataset)
    generator.add_augmentations(augmentations)
    # torch types a DataLoader by its dataset's item type (numpy arrays here),
    # but the default collate function stacks the items into tensors.
    loader = DataLoader(
        generator,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        worker_init_fn=worker_seeding,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available(),
        # fork shares the preloaded waveforms with the workers (copy-on-write)
        # instead of copying them; Python 3.14 no longer defaults to it.
        multiprocessing_context="fork",
    )
    return cast(DataLoader[Batch], loader)
