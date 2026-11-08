import tempfile
from collections.abc import Iterator
from pathlib import Path

import mlflow
import pytest

from seisbench_trainer.config import TrainConfig
from seisbench_trainer.data.synthetic import write_synthetic_dataset

N_TRACES = 120


@pytest.fixture(scope="session")
def synthetic_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small synthetic dataset, written once per test session."""
    path = tmp_path_factory.mktemp("synthetic")
    write_synthetic_dataset(path, n_traces=N_TRACES, seed=1)
    return path


@pytest.fixture
def isolated_tempdir(tmp_path: Path) -> Iterator[Path]:
    """Point ``tempfile.gettempdir()`` (synthetic dataset cache) at tmp_path."""
    previous = tempfile.tempdir
    tempfile.tempdir = str(tmp_path)
    yield tmp_path
    tempfile.tempdir = previous


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A fresh SQLite MLflow store for each test."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    yield uri


def small_config(
    synthetic_path: Path, tmp_path: Path, epochs: int, **sections: object
) -> TrainConfig:
    return TrainConfig.model_validate(
        {
            "evaluation": {"baseline": None},
            "data": {
                "datasets": [{"type": "local", "path": str(synthetic_path)}],
                "batch_size": 16,
                "num_workers": 1,
            },
            "training": {
                "epochs": epochs,
                "accelerator": "cpu",
                "checkpoint_dir": str(tmp_path / "checkpoints"),
            },
            "tracking": {"tags": {"purpose": "test"}},
        }
        | sections
    )
