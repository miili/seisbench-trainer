from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from seisbench_trainer.cli import app
from seisbench_trainer.config import TrainConfig, dump_config, load_config
from seisbench_trainer.data import LocalDataset, SeisBenchDataset
from seisbench_trainer.data.preprocessing import Filter, Resplit
from seisbench_trainer.models import PhaseNetConfig
from seisbench_trainer.training.optim import (
    AdamConfig,
    AdamWConfig,
    CosineConfig,
    ExponentialConfig,
    PlateauConfig,
)
from seisbench_trainer.units import seconds_to_samples

PICK = {"type": "pick_window", "probability": 0.5}
CROP = {"type": "random_window"}
DROPOUT = {"type": "channel_dropout", "probability": 0.3}
NOISE = {"type": "real_noise", "probability": 0.5}


def error_types(raw: dict[str, Any]) -> set[str]:
    with pytest.raises(ValidationError) as info:
        TrainConfig.model_validate(raw)
    return {str(error["type"]) for error in info.value.errors()}


def test_default_round_trip(tmp_path: Path) -> None:
    config = TrainConfig()
    path = tmp_path / "run.json"
    path.write_text(dump_config(config))
    assert load_config(path) == config


@pytest.mark.parametrize(
    ("key", "tag", "expected"),
    [
        ("optimizer", "adam", AdamConfig),
        ("optimizer", "adamw", AdamWConfig),
        ("scheduler", "cosine", CosineConfig),
        ("scheduler", "plateau", PlateauConfig),
        ("scheduler", "exponential", ExponentialConfig),
    ],
)
def test_optim_union_members_parse(key: str, tag: str, expected: type[object]) -> None:
    training = {key: {"type": tag}, "early_stopping_patience": None}
    config = TrainConfig.model_validate({"training": training})
    assert isinstance(getattr(config.training, key), expected)


def test_model_union_parses() -> None:
    config = TrainConfig.model_validate(
        {"model": {"type": "PhaseNet", "pretrained": "original"}}
    )
    assert config.model == PhaseNetConfig(pretrained="original")


def test_combined_datasets() -> None:
    config = TrainConfig.model_validate(
        {
            "data": {
                "datasets": [
                    {"type": "seisbench", "name": "ETHZ"},
                    {
                        "type": "local",
                        "path": "/data/x",
                        "query": "source_magnitude > 1",
                    },
                ]
            }
        }
    )
    assert [type(d) for d in config.data.datasets] == [SeisBenchDataset, LocalDataset]


def dataset_steps(*steps: dict[str, Any]) -> dict[str, Any]:
    """A config whose one dataset has the given preprocessing steps."""
    dataset = {"type": "seisbench", "name": "ETHZ", "preprocessing": list(steps)}
    return {"data": {"datasets": [dataset]}}


def test_preprocessing_steps_parse() -> None:
    config = TrainConfig.model_validate(
        dataset_steps(
            {"type": "resplit", "fractions": {"train": 0.8, "dev": 0.2}},
            {"type": "filter", "query": "trace_category == 'noise'"},
        )
    )
    resplit, filter_ = config.data.datasets[0].preprocessing
    assert isinstance(resplit, Resplit)
    assert (resplit.seed, resplit.group_by) == (0, "source_id")
    assert isinstance(filter_, Filter)
    assert filter_.splits == ["train", "dev", "test"]
    assert TrainConfig().data.datasets[0].preprocessing == []


@pytest.mark.parametrize(
    ("monitor", "mode"),
    [
        ("val_loss", "min"),
        ("det/f1_mean", "max"),
        ("det_P/f1", "max"),
        ("det_S/f1", "max"),
    ],
)
def test_monitor_mode_follows_metric(monitor: str, mode: str) -> None:
    config = TrainConfig.model_validate({"training": {"monitor": monitor}})
    assert config.training.monitor_mode == mode


def test_loss_defaults() -> None:
    loss = TrainConfig().model.loss
    assert loss.from_logits
    assert loss.weights == {"N": 0.5}


def test_defaults_find_the_run_length() -> None:
    training = TrainConfig().training
    assert training.scheduler == PlateauConfig()
    assert training.early_stopping_patience == 15
    assert training.scheduler.patience < training.early_stopping_patience
    assert training.min_delta == 0.002
    assert training.epochs == 100


def test_cosine_needs_a_fixed_length() -> None:
    cosine = {"scheduler": {"type": "cosine"}}
    with pytest.raises(ValidationError, match="early_stopping_patience to null"):
        TrainConfig.model_validate({"training": cosine})
    fixed = cosine | {"early_stopping_patience": None}
    assert TrainConfig.model_validate({"training": fixed}).training.scheduler


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"model": {"type": "EQT"}}, "union_tag_invalid"),
        ({"training": {"optimizer": {"type": "sgd"}}}, "union_tag_invalid"),
        ({"data": {"datasets": [{"type": "ftp"}]}}, "union_tag_invalid"),
        ({"data": {"datasets": [{"name": "ETHZ"}]}}, "union_tag_not_found"),
        ({"trainig": {}}, "extra_forbidden"),
        ({"training": {"lr": 1e-3}}, "extra_forbidden"),
        ({"data": {"num_workers": 0}}, "greater_than"),
        ({"data": {"datasets": []}}, "too_short"),
        ({"data": {"component_order": "zne"}}, "string_pattern_mismatch"),
        ({"data": {"component_order": "ZZN"}}, "value_error"),
        (
            {"augmentation": {"train": [PICK | {"before_s": 90.0}, CROP]}},
            "value_error",
        ),
        ({"augmentation": {"train": [PICK]}}, "value_error"),
        ({"augmentation": {"train": [CROP, CROP]}}, "value_error"),
        ({"augmentation": {"train": [CROP, PICK]}}, "value_error"),
        ({"augmentation": {"train": [DROPOUT, CROP]}}, "value_error"),
        (
            {"augmentation": {"train": [CROP, DROPOUT | {"probability": -0.1}]}},
            "greater_than_equal",
        ),
        (
            {
                "augmentation": {
                    "train": [
                        CROP,
                        NOISE | {"level": {"type": "peak", "ratio": [0.5, 0.1]}},
                    ]
                }
            },
            "value_error",
        ),
        (
            {
                "augmentation": {
                    "train": [
                        CROP,
                        NOISE | {"level": {"type": "std", "ratio": [0.0, 2.0]}},
                    ]
                }
            },
            "value_error",
        ),
        ({"augmentation": {"train": [{"type": "pick_window"}, CROP]}}, "missing"),
        (
            {"augmentation": {"train": [CROP, {"type": "add_gap", "probability": 2}]}},
            "less_than_equal",
        ),
        ({"augmentation": {"train": [CROP, {"type": "mixup"}]}}, "union_tag_invalid"),
        ({"evaluation": {"min_peak_height": 1.5}}, "less_than"),
        (dataset_steps({"type": "resplit"}), "missing"),
        (
            dataset_steps({"type": "resplit", "fractions": {"train": 0.5}}),
            "value_error",
        ),
        (
            dataset_steps(
                {"type": "resplit", "fractions": {"train": 1.5, "dev": -0.5}}
            ),
            "less_than_equal",
        ),
        (
            dataset_steps({"type": "resplit", "fractions": {"val": 1.0}}),
            "literal_error",
        ),
        (dataset_steps({"type": "filter"}), "missing"),
        (dataset_steps({"type": "filter", "query": "x", "splits": []}), "too_short"),
        (
            dataset_steps({"type": "filter", "query": "x", "splits": ["dev", "dev"]}),
            "value_error",
        ),
        (dataset_steps({"type": "shuffle"}), "union_tag_invalid"),
        (
            {"model": {"type": "PhaseNet", "loss": {"type": "dice"}}},
            "union_tag_invalid",
        ),
        (
            {
                "model": {
                    "type": "PhaseNet",
                    "loss": {"type": "cross_entropy", "weights": {"P": -1.0}},
                }
            },
            "greater_than_equal",
        ),
        (
            {
                "model": {
                    "type": "PhaseNet",
                    "loss": {"type": "cross_entropy", "pick_mask_s": 0.5},
                }
            },
            "extra_forbidden",
        ),
        ({"training": {"monitor": "pick_P/mae_s"}}, "literal_error"),
        ({"training": {"selection_mode": "max"}}, "extra_forbidden"),
    ],
)
def test_invalid_configs_rejected(raw: dict[str, Any], expected: str) -> None:
    assert expected in error_types(raw)


def test_config_is_frozen() -> None:
    config = TrainConfig()
    with pytest.raises(ValidationError):
        config.training.epochs = 1


@pytest.mark.parametrize(
    ("seconds", "rate", "samples"),
    [(30.01, 100.0, 3001), (0.1, 100.0, 10), (2.0, 50.0, 100), (0.0, 100.0, 0)],
)
def test_seconds_to_samples(seconds: float, rate: float, samples: int) -> None:
    assert seconds_to_samples(seconds, rate) == samples


def test_cli_config_prints_valid_default() -> None:
    result = CliRunner().invoke(app, ["config"])
    assert result.exit_code == 0
    assert TrainConfig.model_validate_json(result.output) == TrainConfig()


def test_cli_config_schema() -> None:
    result = CliRunner().invoke(app, ["config", "--schema"])
    assert result.exit_code == 0
    schema = json.loads(result.output)
    assert {"PhaseNetConfig", "SeisBenchDataset", "AdamConfig"} <= set(schema["$defs"])


@pytest.mark.parametrize(
    "path",
    sorted((Path(__file__).parents[1] / "configs").glob("*.json")),
    ids=lambda path: path.name,
)
def test_example_configs_are_valid(path: Path) -> None:
    load_config(path)


def test_every_config_field_is_documented() -> None:
    # the descriptions are the parameter reference (docs/reference/config.md)
    schema = TrainConfig.model_json_schema()
    undocumented = []
    for name, model in [("TrainConfig", schema), *schema["$defs"].items()]:
        if model.get("type") == "object" and not model.get("description"):
            undocumented.append(f"{name} (class docstring)")
        for field, prop in model.get("properties", {}).items():
            # a union member's type tag is documented by its class docstring
            if field != "type" and not prop.get("description"):
                undocumented.append(f"{name}.{field}")
    assert not undocumented, f"add descriptions: {undocumented}"
