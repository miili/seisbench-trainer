"""Load, split, concatenate and preload datasets; build DataLoaders."""

from __future__ import annotations

import logging
import operator
from functools import reduce
from typing import NamedTuple, cast

import seisbench.data as sbd
import seisbench.generate as sbg
import torch
from seisbench.util import worker_seeding
from torch.utils.data import DataLoader

from seisbench_trainer.data.config import DataConfig
from seisbench_trainer.data.phases import Augmentation

logger = logging.getLogger(__name__)

AnyDataset = sbd.WaveformDataset | sbd.MultiWaveformDataset
Batch = dict[str, torch.Tensor]


class Splits(NamedTuple):
    train: AnyDataset
    dev: AnyDataset
    test: AnyDataset


def load_splits(config: DataConfig, preload: bool = True) -> Splits:
    """Load every dataset, split each one, and concatenate the splits by type."""
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
    splits = Splits(train, dev, test)
    for name, split in splits._asdict().items():
        logger.info("%s split: %d traces", name, len(split))
        if preload:
            split.preload_waveforms(pbar=True)
    return splits


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
