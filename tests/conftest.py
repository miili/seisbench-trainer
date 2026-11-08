import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

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
