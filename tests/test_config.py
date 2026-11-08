import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from seisbench_trainer.cli import app
from seisbench_trainer.config import TrainConfig, dump_config, load_config
from seisbench_trainer.data import LocalDataset, SeisBenchDataset
from seisbench_trainer.models import PhaseNetConfig
from seisbench_trainer.training.optim import (
    AdamConfig,
    AdamWConfig,
    CosineConfig,
    PlateauConfig,
)
from seisbench_trainer.units import seconds_to_samples


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
    ],
)
def test_optim_union_members_parse(key: str, tag: str, expected: type[object]) -> None:
    config = TrainConfig.model_validate({"training": {key: {"type": tag}}})
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


def test_scheduler_defaults_to_none() -> None:
    assert TrainConfig().training.scheduler is None


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
        ({"augmentation": {"pick_context_before_s": 90.0}}, "value_error"),
        ({"evaluation": {"min_peak_height": 1.5}}, "less_than"),
    ],
)
def test_invalid_configs_rejected(raw: dict[str, Any], expected: str) -> None:
    assert expected in error_types(raw)


def test_config_is_frozen() -> None:
    config = TrainConfig()
    with pytest.raises(ValidationError):
        config.training.epochs = 1  # pyright: ignore[reportAttributeAccessIssue]


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
