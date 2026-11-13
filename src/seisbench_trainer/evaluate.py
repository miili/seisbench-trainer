"""Test-set and cross-dataset evaluation of trained or pretrained models.

Thresholds are selected on the dev split (at ``evaluation.tolerance_s``) and
applied unchanged to the evaluated split at every test tolerance.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import mlflow
import numpy as np
import pandas as pd
import seisbench.models as sbm
import torch
from mlflow.artifacts import download_artifacts
from mlflow.data.pandas_dataset import PandasDataset, from_pandas
from mlflow.models import get_model_info
from pydantic import TypeAdapter

from seisbench_trainer.config import TrainConfig
from seisbench_trainer.data import DataConfig, DatasetConfig
from seisbench_trainer.data.augmentations import eval_augmentations
from seisbench_trainer.data.loaders import AnyDataset, Splits, load_splits, make_loader
from seisbench_trainer.data.phases import PHASES
from seisbench_trainer.metrics.picks import PickScores, detect_picks, pick_accuracy
from seisbench_trainer.metrics.scores import (
    residual_summary,
    summarize,
    threshold_curve,
)
from seisbench_trainer.models import ModelConfig
from seisbench_trainer.tracking.run import TRAIN_CONFIG
from seisbench_trainer.units import seconds_to_samples

logger = logging.getLogger(__name__)

Split = Literal["train", "dev", "test"]
RUN_ID = re.compile(r"[0-9a-f]{32}")


@dataclass(frozen=True)
class LoadedModel:
    config: ModelConfig
    model: sbm.WaveformModel
    name: str
    model_id: str | None = None
    """MLflow LoggedModel ID; evaluation metrics are linked to it."""
    source: TrainConfig | None = None
    """Config of the training run, if the model comes from MLflow."""


@dataclass(frozen=True)
class EvaluationRow:
    model: str
    phase: str
    tolerance_s: float
    threshold: float
    precision: float
    recall: float
    f1: float
    mae_s: float
    median_s: float
    rms_s: float
    n_picks: int
    n_false_picks: int


def _model_id(ref: str) -> str:
    if RUN_ID.fullmatch(ref):
        # search the run's experiment: the default is only the active one
        experiment_id = mlflow.get_run(ref).info.experiment_id
        models = mlflow.search_logged_models(
            experiment_ids=[experiment_id],
            filter_string=f"source_run_id = '{ref}'",
            output_format="list",
        )
        if not models:
            raise ValueError(f"run {ref} has no logged model")
        return models[0].model_id
    name = ref.removeprefix("models:/")
    if name.startswith("m-"):
        return name
    return get_model_info(ref).model_id


def load_model(ref: str) -> LoadedModel:
    """Load a model by reference.

    ``ref`` is an MLflow model (``models:/<model_id>``,
    ``models:/<name>@<alias>``, ``models:/<name>/<version>``), the ID of a
    training run, or ``<Type>:<weights>`` with a SeisBench pretrained name or
    a local model path, e.g. ``PhaseNet:original``.
    """
    if ref.startswith("models:/") or RUN_ID.fullmatch(ref):
        model_id = _model_id(ref)
        with TemporaryDirectory() as tmp:
            path = Path(download_artifacts(f"models:/{model_id}", dst_path=tmp))
            extra = path / "extra_files"
            source = TrainConfig.model_validate_json((extra / TRAIN_CONFIG).read_text())
            export = next(extra.glob("*.pt")).with_suffix("")
            weights = source.model.model_copy(update={"pretrained": str(export)})
            model = weights.load_pretrained()
        return LoadedModel(source.model, model, export.name, model_id, source)

    model_type, separator, weights = ref.partition(":")
    if not separator:
        raise ValueError(f"model reference {ref!r} is not 'Type:weights'")
    config = TypeAdapter(ModelConfig).validate_python(
        {"type": model_type, "pretrained": weights}
    )
    return LoadedModel(config, config.load_pretrained(), ref)


def parse_dataset(spec: str) -> DatasetConfig:
    """A dataset from the CLI: JSON config, local directory or seisbench name."""
    from seisbench_trainer.data import LocalDataset, SeisBenchDataset

    if spec.lstrip().startswith("{"):
        return TypeAdapter(DatasetConfig).validate_json(spec)
    if Path(spec).is_dir():
        return LocalDataset(path=Path(spec))
    return SeisBenchDataset(name=spec)


def _collect(
    loaded: LoadedModel,
    dataset: AnyDataset,
    settings: TrainConfig,
    data_order: str,
    tolerances_s: list[float],
) -> dict[str, PickScores]:
    """Run the model over a split. Keys ``pick_<phase>``, ``det_<phase>@<tol>``."""
    evaluation = settings.evaluation
    rate = settings.data.sampling_rate
    model, config = loaded.model, loaded.config
    window = (
        config.window_samples(model)
        if settings.augmentation.window_s is None
        else seconds_to_samples(settings.augmentation.window_s, rate)
    )
    reorder = [data_order.index(c) for c in model.component_order]
    loader = make_loader(
        dataset,
        eval_augmentations(settings.augmentation, window, settings.training.seed, rate),
        settings.data,
        shuffle=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    parts: dict[str, list[PickScores]] = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            curves = config.phase_probabilities(
                model, config.forward(model, batch["X"][:, reorder].to(device))
            )
            picks = batch["pick_samples"].numpy()
            unread = batch["label_mask"].numpy() == 0
            for index, phase in enumerate(PHASES):
                probs = curves[phase].float().cpu().numpy()
                parts[f"pick_{phase}"].append(
                    pick_accuracy(
                        probs,
                        picks[:, index],
                        seconds_to_samples(evaluation.pick_window_s, rate),
                        seconds_to_samples(evaluation.edge_mask_s, rate),
                        unread[:, index],
                    )
                )
                for tolerance in tolerances_s:
                    parts[f"det_{phase}@{tolerance:g}"].append(
                        detect_picks(
                            probs,
                            picks[:, index],
                            evaluation.min_peak_height,
                            seconds_to_samples(evaluation.min_peak_distance_s, rate),
                            seconds_to_samples(tolerance, rate),
                            unread[:, index],
                        )
                    )
    return {key: PickScores.concatenate(value) for key, value in parts.items()}


def _dataset_entity(dataset: AnyDataset, name: str) -> PandasDataset:
    """Lightweight MLflow dataset: name plus a digest of the trace names."""
    return from_pandas(dataset.metadata[["trace_name"]], name=name)


def evaluate(
    loaded: LoadedModel,
    settings: TrainConfig,
    split: Split = "test",
    splits: Splits | None = None,
    datasets: list[DatasetConfig] | None = None,
    prefix: str = "",
) -> list[EvaluationRow]:
    """Evaluate a model on one split and log the results to the active run.

    ``settings`` gives the data, augmentation and evaluation parameters.
    Already loaded ``splits`` (in ``settings.data.component_order``) are reused;
    otherwise ``datasets`` (default: ``settings.data.datasets``) are loaded in
    the model's own component order. Metrics are named
    ``{prefix}{split}_pick_{phase}/...`` and
    ``{prefix}{split}_det_{phase}/tol{t}s/...`` and linked to the MLflow model
    and dataset.
    """
    model = loaded.model
    if model.sampling_rate != settings.data.sampling_rate:
        raise ValueError(
            f"{loaded.name} expects {model.sampling_rate} Hz, "
            f"the data has {settings.data.sampling_rate} Hz"
        )
    data_order = settings.data.component_order
    if splits is None:
        data = DataConfig.model_validate(
            settings.data.model_dump()
            | {
                "datasets": [
                    d.model_dump() for d in (datasets or settings.data.datasets)
                ],
                "component_order": model.component_order,
            }
        )
        settings = settings.model_copy(update={"data": data})
        data_order = model.component_order
        splits = load_splits(data, preload=False)
        for needed in {"dev", split}:
            getattr(splits, needed).preload_waveforms(pbar=True)

    evaluation = settings.evaluation
    rate = settings.data.sampling_rate
    tolerances = evaluation.test_tolerances_s
    dev = _collect(loaded, splits.dev, settings, data_order, [evaluation.tolerance_s])
    selection_key = f"@{evaluation.tolerance_s:g}"
    thresholds = {
        phase: summarize(dev[f"det_{phase}{selection_key}"], rate, [])["threshold"]
        for phase in PHASES
    }
    results = _collect(loaded, getattr(splits, split), settings, data_order, tolerances)

    label = "+".join(source.label() for source in settings.data.datasets)
    dataset = _dataset_entity(getattr(splits, split), f"{label}/{split}")
    mlflow.log_input(dataset, context=f"{prefix}{split}")
    metrics: dict[str, float] = {}
    rows: list[EvaluationRow] = []
    for phase in PHASES:
        base = f"{prefix}{split}_"
        pick = summarize(results[f"pick_{phase}"], rate, tolerances)
        metrics |= {f"{base}pick_{phase}/{k}": v for k, v in pick.items()}
        threshold = thresholds[phase]
        metrics[f"{base}det_{phase}/threshold"] = threshold
        for tolerance in tolerances:
            scores = results[f"det_{phase}@{tolerance:g}"]
            curve = threshold_curve(scores, np.array([threshold]))
            found = scores.residuals[scores.scores >= threshold] / rate
            stats = residual_summary(found, [])
            row = EvaluationRow(
                model=loaded.name,
                phase=phase,
                tolerance_s=tolerance,
                threshold=threshold,
                precision=float(curve.precision[0]),
                recall=float(curve.recall[0]),
                f1=float(curve.f1[0]),
                mae_s=stats["mae_s"],
                median_s=stats["median_s"],
                rms_s=stats["rms_s"],
                n_picks=int(scores.scores.size),
                n_false_picks=int((scores.negative_scores >= threshold).sum()),
            )
            rows.append(row)
            key = f"{base}det_{phase}/tol{tolerance:g}s"
            for name in ("precision", "recall", "f1", "mae_s", "median_s", "rms_s"):
                metrics[f"{key}/{name}"] = float(getattr(row, name))

    for key, value in metrics.items():
        mlflow.log_metric(key, value, model_id=loaded.model_id, dataset=dataset)
    # CSV, because the MLflow UI renders .csv as a table and .json as raw text.
    mlflow.log_text(
        pd.DataFrame([asdict(row) for row in rows]).to_csv(index=False),
        f"evaluation/{prefix}{split}_summary.csv",
    )
    logger.info("Evaluated %s on %s/%s", loaded.name, label, split)
    return rows
