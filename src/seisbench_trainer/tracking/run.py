"""Run metadata and model export to MLflow."""

import os
import socket
import subprocess
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import numpy as np
import seisbench.models as sbm
from mlflow.pytorch import (  # mlflow leaves its model parameter untyped
    log_model,  # pyright: ignore[reportUnknownVariableType]
)
from pydantic import JsonValue

from seisbench_trainer.config import TrainConfig

TRACKED_PACKAGES = ("seisbench-trainer", "seisbench", "torch", "lightning", "mlflow")
SLURM_VARIABLES = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID")


def dataset_label(config: TrainConfig) -> str:
    return "+".join(source.label() for source in config.data.datasets)


def experiment_name(config: TrainConfig) -> str:
    """Configured experiment, or ``<datasets>/<model>``."""
    return config.tracking.experiment or f"{dataset_label(config)}/{config.model.type}"


def model_name(config: TrainConfig) -> str:
    """SeisBench export name: ``<model>-<datasets>-<variant>``."""
    return f"{config.model.type}-{dataset_label(config)}-{config.model.variant()}"


def flatten(value: JsonValue, prefix: str = "") -> dict[str, str]:
    """Flatten nested dicts and lists to dotted keys, for MLflow params."""
    if isinstance(value, dict):
        items = [(str(key), item) for key, item in value.items()]
    elif isinstance(value, list):
        items = [(str(index), item) for index, item in enumerate(value)]
    else:
        return {prefix: str(value)}
    flat: dict[str, str] = {}
    for key, item in items:
        flat |= flatten(item, f"{prefix}.{key}" if prefix else key)
    return flat


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def read_git_head(start: Path) -> str | None:
    """Commit of the repository containing ``start``, read from ``.git``.

    Fallback for nodes without the git executable (e.g. cluster nodes).
    """
    for directory in (start, *start.parents):
        git_dir = directory / ".git"
        head = git_dir / "HEAD"
        if not head.is_file():
            continue
        content = head.read_text().strip()
        if not content.startswith("ref: "):
            return content  # detached HEAD
        ref = content.removeprefix("ref: ")
        if (git_dir / ref).is_file():
            return (git_dir / ref).read_text().strip()
        packed = git_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text().splitlines():
                if line.endswith(f" {ref}"):
                    return line.split()[0]
        return None
    return None


def run_tags(config: TrainConfig) -> dict[str, str]:
    tags = {
        "model": config.model.type,
        "variant": config.model.variant(),
        "datasets": dataset_label(config),
        "host": socket.gethostname(),
    }
    tags |= {f"version.{name}": version(name) for name in TRACKED_PACKAGES}
    tags |= {
        f"slurm.{name.removeprefix('SLURM_').lower()}": os.environ[name]
        for name in SLURM_VARIABLES
        if name in os.environ
    }
    commit = _git("rev-parse", "HEAD") or read_git_head(Path(__file__).parent)
    if commit is not None:
        tags["git.commit"] = commit
        status = _git("status", "--porcelain")
        tags["git.dirty"] = "unknown" if status is None else str(bool(status))
    return tags | config.tracking.tags


def log_run_metadata(config: TrainConfig) -> None:
    """Log the resolved config (artifact and params) and the run tags."""
    config_dict: JsonValue = config.model_dump(mode="json")
    mlflow.log_dict(config_dict, "config.json")
    mlflow.log_params(flatten(config_dict))
    mlflow.set_tags(run_tags(config))


def log_seisbench_model(
    model: sbm.WaveformModel, config: TrainConfig, window_samples: int, step: int
) -> str:
    """Log the model as an MLflow LoggedModel with its SeisBench export attached.

    Returns the model ID. The SeisBench ``.json``/``.pt`` pair is stored under
    the model's ``extra_files`` and loads with ``sbm.<Model>.load()``.
    """
    if model.component_order != config.data.component_order:
        raise ValueError(
            f"model metadata says {model.component_order!r}, "
            f"but it was trained on {config.data.component_order!r}"
        )
    model.eval()
    channels = len(config.data.component_order)
    input_example = np.zeros((1, channels, window_samples), dtype=np.float32)
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / model_name(config)
        model.save(path)
        files = [str(path.with_suffix(suffix)) for suffix in (".json", ".pt")]
        requirements = [f"{name}=={version(name)}" for name in ("seisbench", "torch")]
        info = log_model(
            model,
            name="model",
            input_example=input_example,
            extra_files=files,
            pip_requirements=requirements,  # skips slow requirement inference
            step=step,
        )
    return info.model_id
