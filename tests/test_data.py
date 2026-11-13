from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import seisbench.data as sbd
import seisbench.generate as sbg
from conftest import N_TRACES
from numpy.typing import NDArray

from seisbench_trainer.data import (
    DataConfig,
    LocalDataset,
    SeisBenchDataset,
    SyntheticDataset,
)
from seisbench_trainer.data.augmentations import (
    ZeroNonFinite,
    eval_augmentations,
    train_augmentations,
)
from seisbench_trainer.data.config import AugmentationConfig
from seisbench_trainer.data.loaders import load_splits, make_loader
from seisbench_trainer.data.phases import (
    PHASE_COLUMNS,
    PHASES,
    Augmentation,
)
from seisbench_trainer.data.steps import StepContext

WINDOW = 3001
SAMPLING_RATE = 100.0


def labeller() -> Augmentation:
    return sbg.ProbabilisticLabeller(
        label_columns=PHASE_COLUMNS, sigma=10, dim=0, model_labels="PSN"
    )


def eval_generator(
    dataset: sbd.WaveformDataset | sbd.MultiWaveformDataset, seed: int = 0
) -> sbg.GenericGenerator:
    generator = sbg.GenericGenerator(dataset)
    generator.add_augmentations(
        [*eval_augmentations(AugmentationConfig(), WINDOW, seed), labeller()]
    )
    return generator


def local(path: Path, order: str = "ZNE") -> sbd.WaveformDataset:
    return LocalDataset(path=path).load(order, SAMPLING_RATE)


def test_synthetic_dataset_contents(synthetic_path: Path) -> None:
    metadata = local(synthetic_path).metadata
    assert len(metadata) == N_TRACES
    assert set(metadata["split"]) == {"train", "dev", "test"}
    events = metadata["trace_category"] == "earthquake"
    assert 0 < (~events).sum() < N_TRACES // 2
    assert metadata.loc[~events, "trace_p_arrival_sample"].isna().all()
    picks = metadata.loc[events]
    assert (picks["trace_p_arrival_sample"] < picks["trace_s_arrival_sample"]).all()


def test_synthetic_source_is_cached(isolated_tempdir: Path) -> None:
    source = SyntheticDataset(n_traces=20)
    path = source.ensure_written()
    assert path.is_relative_to(isolated_tempdir)
    assert (path / "metadata.csv").exists()
    assert source.ensure_written() == path
    assert SyntheticDataset(n_traces=20, seed=1).path() != path
    assert SyntheticDataset(n_traces=20, query="x > 1").path() == path
    assert len(source.load("ZNE", SAMPLING_RATE)) == 20


def test_component_order_is_applied(synthetic_path: Path) -> None:
    """P is strongest on Z, wherever Z sits in the requested order."""
    for order in ("ZNE", "ENZ"):
        dataset = local(synthetic_path, order)
        metadata = dataset.metadata
        index = int(np.flatnonzero(metadata["trace_category"] == "earthquake")[0])
        waveforms, _ = dataset.get_sample(index)
        p = int(metadata["trace_p_arrival_sample"].iloc[index])
        p_energy = np.abs(waveforms[:, p : p + 100]).max(axis=-1)
        assert int(np.argmax(p_energy)) == order.index("Z")


def test_query_filters_metadata(synthetic_path: Path) -> None:
    source = LocalDataset(path=synthetic_path, query="trace_category == 'earthquake'")
    dataset = source.load("ZNE", SAMPLING_RATE)
    assert 0 < len(dataset) < N_TRACES
    assert (dataset.metadata["trace_category"] == "earthquake").all()


def test_query_must_give_boolean_mask(synthetic_path: Path) -> None:
    source = LocalDataset(path=synthetic_path, query="source_magnitude + 1")
    with pytest.raises(ValueError, match="boolean mask"):
        source.load("ZNE", SAMPLING_RATE)


@pytest.mark.parametrize("name", ["NoSuchDataset", "WaveformDataset"])
def test_unknown_seisbench_dataset(name: str) -> None:
    with pytest.raises(ValueError, match="not a dataset class"):
        SeisBenchDataset(name=name).load("ZNE", SAMPLING_RATE)


def test_load_splits_concatenates_datasets(synthetic_path: Path) -> None:
    single = load_splits(DataConfig(datasets=[LocalDataset(path=synthetic_path)]))
    double = load_splits(
        DataConfig(datasets=[LocalDataset(path=synthetic_path)] * 2), preload=False
    )
    for one, two in zip(single, double, strict=True):
        assert len(one) > 0
        assert len(two) == 2 * len(one)
    assert sum(len(split) for split in single) == N_TRACES


def test_train_loader_batches(synthetic_path: Path) -> None:
    splits = load_splits(DataConfig(datasets=[LocalDataset(path=synthetic_path)]))
    context = StepContext(SAMPLING_RATE, WINDOW, "ZNE", splits.train)
    augmentations = [*train_augmentations(AugmentationConfig(), context), labeller()]
    config = DataConfig(batch_size=16, num_workers=1)
    batch = next(iter(make_loader(splits.train, augmentations, config, shuffle=True)))
    assert tuple(batch["X"].shape) == (16, 3, WINDOW)
    assert tuple(batch["y"].shape) == (16, 3, WINDOW)
    assert str(batch["X"].dtype) == "torch.float32"


def items(generator: sbg.GenericGenerator) -> list[dict[str, NDArray[Any]]]:
    return [generator[index] for index in range(len(generator))]


def test_pick_samples_match_labels(synthetic_path: Path) -> None:
    samples = items(eval_generator(local(synthetic_path)))
    n_checked = 0
    for sample in samples:
        assert sample["pick_samples"].shape == (len(PHASES), 4)
        for row, phase in enumerate(PHASES):
            channel = "PSN".index(phase)
            for pick in sample["pick_samples"][row]:
                if np.isfinite(pick) and 20 <= pick < WINDOW - 20:
                    peak = int(np.argmax(sample["y"][channel]))
                    assert abs(peak - pick) <= 1
                    n_checked += 1
    assert n_checked > N_TRACES // 2


def test_noise_traces_have_no_picks(synthetic_path: Path) -> None:
    dataset = local(synthetic_path)
    noise = np.flatnonzero(dataset.metadata["trace_category"] == "noise")
    generator = eval_generator(dataset)
    for index in noise:
        assert np.isnan(generator[int(index)]["pick_samples"]).all()


def test_eval_windows_are_deterministic(synthetic_path: Path) -> None:
    dataset = local(synthetic_path)
    first, second = eval_generator(dataset), eval_generator(dataset)
    other_seed = eval_generator(dataset, seed=1)
    differs = False
    for index in range(10):
        np.testing.assert_array_equal(first[index]["X"], second[index]["X"])
        differs |= not np.array_equal(first[index]["X"], other_seed[index]["X"])
    assert differs


def test_eval_windows_are_not_centred(synthetic_path: Path) -> None:
    """The first pick is spread over the window, not fixed at its centre."""
    samples = items(eval_generator(local(synthetic_path)))
    first_p = np.array([s["pick_samples"][0, 0] for s in samples])
    first_p = first_p[np.isfinite(first_p)]
    assert first_p.min() < WINDOW / 4
    assert first_p.max() > 3 * WINDOW / 4


ALL_STEPS: list[dict[str, Any]] = [
    {"type": "pick_window", "probability": 1.0, "length_s": 40.0, "before_s": 20.0},
    {"type": "random_window"},
    {"type": "channel_dropout", "probability": 1.0},
    {"type": "add_gap", "probability": 1.0},
    {"type": "gaussian_noise", "probability": 1.0, "scale": [0.1, 0.2]},
    {"type": "rotate_horizontal", "probability": 1.0},
    {"type": "real_noise", "probability": 1.0, "level": {"type": "peak"}},
    {"type": "real_noise", "probability": 1.0, "level": {"type": "std"}},
]


def train_generator(
    dataset: sbd.WaveformDataset, steps: list[dict[str, Any]], order: str = "ZNE"
) -> sbg.GenericGenerator:
    config = AugmentationConfig.model_validate({"train": steps})
    context = StepContext(SAMPLING_RATE, WINDOW, order, dataset)
    generator = sbg.GenericGenerator(dataset)
    generator.add_augmentations([*train_augmentations(config, context), labeller()])
    return generator


def test_every_train_step_builds_and_keeps_shape(synthetic_path: Path) -> None:
    generator = train_generator(local(synthetic_path), ALL_STEPS)
    for index in range(10):
        sample = generator[index]
        assert sample["X"].shape == (3, WINDOW)
        assert np.isfinite(sample["X"]).all()


def test_step_probability_is_applied(synthetic_path: Path) -> None:
    dataset = local(synthetic_path)
    noise = {"type": "gaussian_noise", "scale": [0.5, 0.5]}

    def samples(*extra: dict[str, Any]) -> list[NDArray[Any]]:
        generator = train_generator(dataset, [{"type": "random_window"}, *extra])
        crops: list[NDArray[Any]] = []
        for index in range(20):
            np.random.seed(index)  # the crop is drawn first: same in every pipeline
            crops.append(generator[index]["X"])
        return crops

    plain = samples()
    never = samples(noise | {"probability": 0.0})
    always = samples(noise | {"probability": 1.0})
    assert all(np.array_equal(a, b) for a, b in zip(plain, never, strict=True))
    assert not any(np.array_equal(a, b) for a, b in zip(plain, always, strict=True))


def test_non_finite_samples_become_zero(synthetic_path: Path) -> None:
    x = np.ones((3, 100))
    x[1] = np.nan  # a missing component
    x[0, 10:20] = np.inf
    state = {"X": (x, {"trace_name": "a"})}
    ZeroNonFinite()(state)
    out, metadata = state["X"]
    assert np.isfinite(out).all()
    assert (out[1] == 0).all() and (out[0, 10:20] == 0).all()
    assert (out[2] == 1).all() and metadata == {"trace_name": "a"}
    assert np.isnan(x[1]).all()  # input untouched

    finite = np.ones((3, 100))
    state = {"X": (finite, None)}
    ZeroNonFinite()(state)
    assert state["X"][0] is finite  # no copy when there is nothing to fill

    # first in both pipelines: perturbations and normalization see no NaN
    context = StepContext(SAMPLING_RATE, WINDOW, "ZNE", local(synthetic_path))
    train = train_augmentations(AugmentationConfig(), context)
    evaluation = eval_augmentations(AugmentationConfig(), WINDOW, 0)
    assert isinstance(train[0], ZeroNonFinite)
    assert isinstance(evaluation[0], ZeroNonFinite)


def test_noisy_validation_windows_are_seeded(synthetic_path: Path) -> None:
    from seisbench_trainer.data.steps import NoisyValidation

    dataset = local(synthetic_path)
    noise = NoisyValidation().build(dataset, WINDOW, "ZNE")
    assert noise is not None

    def generator(with_noise: bool) -> sbg.GenericGenerator:
        generator = sbg.GenericGenerator(dataset)
        pipeline = eval_augmentations(
            AugmentationConfig(), WINDOW, 0, noise if with_noise else None
        )
        generator.add_augmentations([*pipeline, labeller()])
        return generator

    clean, first, second = generator(False), generator(True), generator(True)
    changed = unchanged = 0
    for index in range(30):
        np.random.seed(index)  # the global state must not matter
        a = first[index]
        np.random.seed(1000 + index)
        b = second[index]
        np.testing.assert_array_equal(a["X"], b["X"])  # same noise every pass
        has_pick = bool(np.isfinite(a["pick_samples"]).any())
        same_as_clean = np.array_equal(a["X"], clean[index]["X"])
        assert same_as_clean != has_pick  # only windows with a pick get noise
        changed += has_pick
        unchanged += not has_pick
    assert changed > 0 and unchanged > 0


def test_noisy_validation_without_noise_traces(synthetic_path: Path) -> None:
    from seisbench_trainer.data.steps import NoisyValidation

    step = NoisyValidation(query='trace_category == "explosion"')
    assert step.build(local(synthetic_path), WINDOW, "ZNE") is None
