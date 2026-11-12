from __future__ import annotations

from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pytest
import seisbench.generate as sbg
import seisbench.models as sbm
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau

from seisbench_trainer.config import TrainConfig
from seisbench_trainer.data import AugmentationConfig, DataConfig, LocalDataset
from seisbench_trainer.data.augmentations import eval_augmentations
from seisbench_trainer.data.loaders import Batch, load_splits, make_loader
from seisbench_trainer.models import PhaseNetConfig
from seisbench_trainer.training.module import PickerModule
from seisbench_trainer.training.optim import (
    AdamConfig,
    AdamWConfig,
    CosineConfig,
    ExponentialConfig,
    PlateauConfig,
)

SAMPLING_RATE = 100.0


@pytest.fixture(scope="session")
def pretrained_original() -> sbm.PhaseNet:
    """SeisBench's pretrained 'original' PhaseNet; skips the test when offline."""
    try:
        return sbm.PhaseNet.from_pretrained("original")
    except Exception as error:  # download failures come in many types
        pytest.skip(f"pretrained weights not available: {error}")


def config_for(path: Path, **training: object) -> TrainConfig:
    return TrainConfig.model_validate(
        {
            "data": DataConfig(
                datasets=[LocalDataset(path=path)], batch_size=8, num_workers=1
            ).model_dump(),
            "training": training,
        }
    )


@pytest.mark.parametrize("order", ["ZNE", "ENZ"])
def test_build_from_scratch(order: str) -> None:
    model = PhaseNetConfig().build(order, SAMPLING_RATE)
    assert model.component_order == order
    assert model.sampling_rate == SAMPLING_RATE
    assert model.in_samples == 3001
    assert sorted(model.labels) == ["N", "P", "S"]


def test_build_pretrained_keeps_weights(pretrained_original: sbm.PhaseNet) -> None:
    order = pretrained_original.component_order
    model = PhaseNetConfig(pretrained="original").build(order, SAMPLING_RATE)
    assert model.component_order == order
    assert model.labels == pretrained_original.labels
    expected = pretrained_original.state_dict()
    for name, weights in model.state_dict().items():
        torch.testing.assert_close(weights, expected[name])


def test_pretrained_component_order_must_match(
    pretrained_original: sbm.PhaseNet,
) -> None:
    assert pretrained_original.component_order == "ENZ"
    with pytest.raises(ValueError, match="'ENZ'"):
        PhaseNetConfig(pretrained="original").build("ZNE", SAMPLING_RATE)


def test_pretrained_sampling_rate_must_match(pretrained_original: sbm.PhaseNet) -> None:
    with pytest.raises(ValueError, match="Hz"):
        PhaseNetConfig(pretrained="original").build("ENZ", 50.0)


def test_forward_returns_probabilities() -> None:
    config = PhaseNetConfig()
    model = config.build("ZNE", SAMPLING_RATE)
    pred = config.forward(model, torch.randn(2, 3, 3001))
    assert pred.shape == (2, 3, 3001)
    torch.testing.assert_close(pred.sum(dim=1), torch.ones(2, 3001))


def test_forward_is_activated_logits() -> None:
    config = PhaseNetConfig()
    model = config.build("ZNE", SAMPLING_RATE)
    model.eval()
    x = torch.randn(2, 3, 3001)
    with torch.no_grad():
        logits = config.forward_logits(model, x)
        torch.testing.assert_close(config.activate(logits), config.forward(model, x))
        assert not torch.allclose(logits.sum(dim=1), torch.ones(2, 3001))


def test_loss_prefers_the_target() -> None:
    target = torch.zeros(1, 3, 100)
    target[0, 2] = 1.0
    target[0, 0, 40:60] = torch.linspace(0, 1, 20)
    target[0, 2] -= target[0, 0]
    batch: Batch = {"y": target}
    loss = PhaseNetConfig().loss.build("PSN")
    matching = torch.log(target + 1e-9)  # softmax of these logits is the target
    assert loss(matching, batch) < loss(torch.zeros_like(target), batch)
    assert loss(matching, batch) >= 0


def test_phase_probabilities_follow_labels() -> None:
    config = PhaseNetConfig()
    model = config.build("ZNE", SAMPLING_RATE)
    pred = torch.arange(3.0).reshape(1, 3, 1).expand(4, 3, 10)
    phases = config.phase_probabilities(model, pred)
    assert set(phases) == {"P", "S"}
    for phase, curve in phases.items():
        assert curve.shape == (4, 10)
        assert torch.all(curve == model.labels.index(phase))


def test_label_augmentations_follow_model_labels(synthetic_path: Path) -> None:
    config = PhaseNetConfig()
    model = config.build("ZNE", SAMPLING_RATE)
    dataset = LocalDataset(path=synthetic_path).load("ZNE", SAMPLING_RATE)
    generator = sbg.GenericGenerator(dataset)
    generator.add_augmentations(
        [
            *eval_augmentations(AugmentationConfig(), 3001, seed=0),
            *config.label_augmentations(model, sigma_samples=10, shape="gaussian"),
        ]
    )
    p_channel = model.labels.index("P")
    for index in range(len(generator)):
        sample = generator[index]
        pick = sample["pick_samples"][0, 0]
        if np.isfinite(pick) and 20 <= pick < 2980:
            assert abs(int(np.argmax(sample["y"][p_channel])) - pick) <= 1
            return
    pytest.fail("no P pick found in the synthetic dataset")


def test_window_samples_override(synthetic_path: Path) -> None:
    config = config_for(synthetic_path)
    assert PickerModule(config).window_samples == 3001
    config = config.model_copy(
        update={"augmentation": AugmentationConfig(window_s=20.0)}
    )
    assert PickerModule(config).window_samples == 2000


@pytest.mark.parametrize(
    ("optimizer", "expected"), [(AdamConfig(), Adam), (AdamWConfig(), AdamW)]
)
def test_configure_optimizers(
    synthetic_path: Path,
    optimizer: AdamConfig | AdamWConfig,
    expected: type[torch.optim.Optimizer],
) -> None:
    config = config_for(
        synthetic_path, optimizer=optimizer.model_dump(), scheduler=None
    )
    assert isinstance(PickerModule(config).configure_optimizers(), expected)
    config = config_for(synthetic_path, optimizer=optimizer.model_dump())
    result = PickerModule(config).configure_optimizers()
    assert isinstance(result, dict)
    assert isinstance(result["optimizer"], expected)
    assert "lr_scheduler" in result


@pytest.mark.parametrize(
    ("scheduler", "expected", "monitor"),
    [
        (CosineConfig(), CosineAnnealingLR, None),
        (PlateauConfig(), ReduceLROnPlateau, "val_loss"),
        (ExponentialConfig(), ExponentialLR, None),
    ],
)
def test_scheduler_build(
    scheduler: CosineConfig | PlateauConfig | ExponentialConfig,
    expected: type[object],
    monitor: str | None,
) -> None:
    optimizer = Adam([torch.nn.Parameter(torch.zeros(1))])
    built = scheduler.build(optimizer, epochs=10, monitor="val_loss", mode="min")
    assert isinstance(built["scheduler"], expected)
    assert built.get("monitor") == monitor


def test_plateau_counts_only_changes_above_min_delta() -> None:
    optimizer = Adam([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
    config = PlateauConfig(patience=1)
    scheduler = config.build(optimizer, 10, "det/f1_mean", "max", min_delta=0.002)
    plateau = scheduler["scheduler"]
    assert isinstance(plateau, ReduceLROnPlateau)
    for f1 in (0.80, 0.801, 0.8015):  # gains below min_delta: no improvement
        plateau.step(f1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.005)


def test_exponential_decay_per_epoch() -> None:
    optimizer = Adam([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
    built = ExponentialConfig(decay=0.9).build(optimizer, 10, "val_loss", "min")
    for _ in range(2):
        optimizer.step()
        built["scheduler"].step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01 * 0.9**2)


def test_training_step_backpropagates(synthetic_path: Path) -> None:
    config = config_for(synthetic_path)
    module = PickerModule(config)
    splits = load_splits(config.data)
    augmentations = [
        *eval_augmentations(config.augmentation, module.window_samples, seed=0),
        *config.model.label_augmentations(module.model, 10, "gaussian"),
    ]
    batch = next(iter(make_loader(splits.train, augmentations, config.data, False)))
    loss = module.loss(config.model.forward_logits(module.model, batch["X"]), batch)
    loss.backward()
    gradients = [p.grad for p in module.parameters() if p.requires_grad]
    assert all(g is not None for g in gradients)
    assert any(float(g.abs().sum()) > 0 for g in gradients if g is not None)


def test_lightning_fit_one_batch(synthetic_path: Path) -> None:
    # a bare Trainer logs no det/f1_mean (that is EvaluationCallback's job),
    # so the plateau default has nothing to watch
    config = config_for(synthetic_path, scheduler=None)
    module = PickerModule(config)
    splits = load_splits(config.data)
    labels = config.model.label_augmentations(module.model, 10, "gaussian")
    augmentations = [
        *eval_augmentations(config.augmentation, module.window_samples, seed=0),
        *labels,
    ]
    train = make_loader(splits.train, augmentations, config.data, shuffle=True)
    dev = make_loader(splits.dev, augmentations, config.data, shuffle=False)
    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, train, dev)
    assert np.isfinite(float(trainer.callback_metrics["train_loss"]))
    assert np.isfinite(float(trainer.callback_metrics["val_loss"]))
