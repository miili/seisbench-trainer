"""Run metadata and model export to MLflow."""

from __future__ import annotations

import os
import re
import socket
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import numpy as np
import seisbench.models as sbm
from mlflow.pytorch import (  # mlflow leaves its model parameter untyped
    log_model,
)
from pydantic import JsonValue

from seisbench_trainer.config import TrainConfig

TRACKED_PACKAGES = ("seisbench-trainer", "seisbench", "torch", "lightning", "mlflow")
EXPERIMENT_KIND = "mlflow.experimentKind"
MODEL_TRAINING = "custom_model_development"
DESCRIPTION = "mlflow.note.content"
SLURM_VARIABLES = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID")


def dataset_label(config: TrainConfig) -> str:
    return "+".join(source.label() for source in config.data.datasets)


def experiment_name(config: TrainConfig) -> str:
    """Configured experiment, or ``<datasets>/<model>``."""
    return config.tracking.experiment or f"{dataset_label(config)}/{config.model.type}"


def set_experiment(name: str) -> str:
    """Activate (or create) an experiment and mark it as model training.

    Without the kind tag, the MLflow 3 UI may show it as a GenAI experiment.
    Returns the experiment ID.
    """
    experiment = mlflow.set_experiment(name)
    if experiment.tags.get(EXPERIMENT_KIND) != MODEL_TRAINING:
        mlflow.set_experiment_tag(EXPERIMENT_KIND, MODEL_TRAINING)
    return experiment.experiment_id


def model_name(config: TrainConfig) -> str:
    """Model name: ``<model>-<datasets>-<variant>``.

    Used for the MLflow LoggedModel and the SeisBench export files, so the
    characters MLflow forbids in model names are replaced.
    """
    name = f"{config.model.type}-{dataset_label(config)}-{config.model.variant()}"
    return re.sub(r"[/:.%\"']", "-", name)


def model_description(
    config: TrainConfig,
    window_samples: int,
    epoch: int,
    score: float | None,
    thresholds: dict[str, float],
) -> str:
    """Markdown description of a trained model, shown in the MLflow UI."""
    data = config.data
    start = (
        "trained from scratch"
        if config.model.pretrained is None
        else f"fine-tuned from `{config.model.pretrained}`"
    )
    if config.model.pretrained is not None and config.model.sampling_rate is not None:
        start += f", its weights run (time-stretched) at {data.sampling_rate:g} Hz"
    selection = config.training.monitor
    best = f"`{selection}` = {score:.4g}" if score is not None else f"`{selection}`"
    return (
        f"{config.model.type} {start} on {dataset_label(config)}.\n\n"
        f"- Input: {data.component_order}, {data.sampling_rate:g} Hz, "
        f"{window_samples} samples ({window_samples / data.sampling_rate:g} s)\n"
        f"- Best epoch {epoch} by {best}\n"
        + "".join(
            f"- `{phase}_threshold` = {value:.3f} (F1-optimal on the validation "
            "set; stored in `default_args`)\n"
            for phase, value in thresholds.items()
        )
        + f"- SeisBench export in `extra_files/`: "
        f'`seisbench.models.{config.model.type}.load("{model_name(config)}")`'
    )


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
    # the seisbench-trainer version (setuptools-scm) names the commit
    return tags | config.tracking.tags


def log_run_metadata(config: TrainConfig) -> None:
    """Log the resolved config (artifact and params) and the run tags."""
    config_dict: JsonValue = config.model_dump(mode="json")
    mlflow.log_dict(config_dict, "config.json")
    mlflow.log_params(flatten(config_dict))
    mlflow.set_tags(run_tags(config))


def log_seisbench_model(
    model: sbm.WaveformModel,
    config: TrainConfig,
    window_samples: int,
    step: int,
    score: float | None = None,
    thresholds: dict[str, float] | None = None,
    figures: dict[str, bytes] | None = None,
) -> str:
    """Log the model as an MLflow LoggedModel with its SeisBench export attached.

    Returns the model ID. The SeisBench ``.json``/``.pt`` pair is stored under
    the model's ``extra_files`` and loads with ``sbm.<Model>.load()``. The
    detection ``thresholds`` go into its ``default_args``; the validation
    ``figures`` (PNG per phase) are stored next to it.
    """
    if model.component_order != config.data.component_order:
        raise ValueError(
            f"model metadata says {model.component_order!r}, "
            f"but it was trained on {config.data.component_order!r}"
        )
    model.eval()
    # seisbench reads "<phase>_threshold" from default_args in classify().
    threshold_args = {f"{p}_threshold": v for p, v in (thresholds or {}).items()}
    model.default_args.update(threshold_args)
    channels = len(config.data.component_order)
    input_example = np.zeros((1, channels, window_samples), dtype=np.float32)
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / model_name(config)
        model.save(path)
        files = [str(path.with_suffix(suffix)) for suffix in (".json", ".pt")]
        for phase, png in (figures or {}).items():
            figure_path = Path(tmp) / f"validation_{phase}_epoch{step}.png"
            figure_path.write_bytes(png)
            files.append(str(figure_path))
        requirements = [f"{name}=={version(name)}" for name in ("seisbench", "torch")]
        description = model_description(
            config, window_samples, step, score, thresholds or {}
        )
        info = log_model(
            model,
            name=model_name(config),
            tags={DESCRIPTION: description, "variant": config.model.variant()},
            params={key: f"{value:.3f}" for key, value in threshold_args.items()},
            input_example=input_example,
            extra_files=files,
            pip_requirements=requirements,  # skips slow requirement inference
            # pt2 (torch.export) files load only on the device they were saved
            # on; a pickled model loads on CPU and GPU nodes alike.
            serialization_format="pickle",
            step=step,
        )
    return info.model_id
