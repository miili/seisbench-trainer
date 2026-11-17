"""HPO config: search space, sampler, pruner and study settings.

Optuna is imported only when a sampler, pruner or parameter is used, so the
config loads without the optional ``hpo`` extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import (
    Field,
    JsonValue,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from seisbench_trainer.schema import StrictModel, UserPath

if TYPE_CHECKING:
    import optuna

Scalar = str | int | float | bool | None

FIXED_PATHS = (
    "data.datasets",
    "data.component_order",
    "data.sampling_rate",
    "augmentation.label_events",
    "training.monitor",
    "hpo",
)
"""Config paths a trial must not change: the preloaded data and the objective."""


class FloatParam(StrictModel):
    """A float drawn from [low, high]."""

    type: Literal["float"] = "float"
    path: str = Field(
        description="Dotted config path, e.g. 'training.optimizer.lr'; list "
        "items by index."
    )
    low: float = Field(description="Smallest value.")
    high: float = Field(description="Largest value.")
    log: bool = Field(
        default=False, description="Sample on a log scale (rates, decays)."
    )
    step: PositiveFloat | None = Field(
        default=None, description="Grid step; None samples continuously."
    )

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.low >= self.high:
            raise ValueError("low must be smaller than high")
        return self

    def suggest(self, trial: optuna.Trial) -> float:
        return trial.suggest_float(
            self.path, self.low, self.high, log=self.log, step=self.step
        )

    def example(self) -> float:
        return self.low


class IntParam(StrictModel):
    """An integer drawn from [low, high]."""

    type: Literal["int"] = "int"
    path: str = Field(description="Dotted config path, e.g. 'data.batch_size'.")
    low: int = Field(description="Smallest value.")
    high: int = Field(description="Largest value.")
    log: bool = Field(default=False, description="Sample on a log scale.")
    step: PositiveInt = Field(default=1, description="Grid step.")

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.low >= self.high:
            raise ValueError("low must be smaller than high")
        return self

    def suggest(self, trial: optuna.Trial) -> int:
        return trial.suggest_int(
            self.path, self.low, self.high, log=self.log, step=self.step
        )

    def example(self) -> int:
        return self.low


class CategoricalParam(StrictModel):
    """One of a list of scalars. Switch a union member by its type field,
    e.g. path 'training.optimizer.type' with choices ['adam', 'adamw']."""

    type: Literal["categorical"] = "categorical"
    path: str = Field(description="Dotted config path.")
    choices: list[Scalar] = Field(min_length=1, description="The candidates.")

    def suggest(self, trial: optuna.Trial) -> Scalar:
        return trial.suggest_categorical(self.path, self.choices)

    def example(self) -> Scalar:
        return self.choices[0]


ParamConfig = Annotated[
    FloatParam | IntParam | CategoricalParam, Field(discriminator="type")
]


class TPEConfig(StrictModel):
    """Tree-structured Parzen estimator (Optuna's default)."""

    type: Literal["tpe"] = "tpe"
    n_startup_trials: NonNegativeInt = Field(
        default=10, description="Random trials before TPE takes over."
    )
    seed: int | None = Field(default=None, description="Sampler seed; None draws one.")

    def build(self) -> optuna.samplers.BaseSampler:
        from optuna.samplers import TPESampler

        return TPESampler(n_startup_trials=self.n_startup_trials, seed=self.seed)


class RandomConfig(StrictModel):
    """Independent random sampling."""

    type: Literal["random"] = "random"
    seed: int | None = Field(default=None, description="Sampler seed; None draws one.")

    def build(self) -> optuna.samplers.BaseSampler:
        from optuna.samplers import RandomSampler

        return RandomSampler(seed=self.seed)


class MedianConfig(StrictModel):
    """Prune a trial below the median of earlier trials at the same epoch."""

    type: Literal["median"] = "median"
    n_startup_trials: NonNegativeInt = Field(
        default=5, description="Trials that are never pruned."
    )
    n_warmup_steps: NonNegativeInt = Field(
        default=5, description="Epochs before a trial can be pruned."
    )

    def build(self) -> optuna.pruners.BasePruner:
        from optuna.pruners import MedianPruner

        return MedianPruner(
            n_startup_trials=self.n_startup_trials,
            n_warmup_steps=self.n_warmup_steps,
        )


class HyperbandConfig(StrictModel):
    """Hyperband: successive halving over several brackets."""

    type: Literal["hyperband"] = "hyperband"
    min_resource: PositiveInt = Field(
        default=1, description="Fewest epochs before a trial can be pruned."
    )
    reduction_factor: int = Field(
        default=3, ge=2, description="Share of trials kept per rung: 1 / factor."
    )

    def build(self) -> optuna.pruners.BasePruner:
        from optuna.pruners import HyperbandPruner

        return HyperbandPruner(
            min_resource=self.min_resource, reduction_factor=self.reduction_factor
        )


SamplerConfig = Annotated[TPEConfig | RandomConfig, Field(discriminator="type")]
PrunerConfig = Annotated[MedianConfig | HyperbandConfig, Field(discriminator="type")]


class HPOConfig(StrictModel):
    """Hyperparameter search with Optuna (the optional hpo extra). Every
    trial is a full training run with the sampled values applied."""

    study_name: str = Field(
        description="Optuna study, shared by all workers; also the MLflow "
        "parent run name (hpo-<study>)."
    )
    storage: UserPath = Field(
        default=Path("optuna.journal"),
        description="Optuna journal file, shared by all workers of the study "
        "(on the shared filesystem on a cluster).",
    )
    n_trials: PositiveInt = Field(
        default=20, description="Total number of trials over all workers."
    )
    sampler: SamplerConfig = Field(
        default_factory=TPEConfig, description="How trials draw values."
    )
    pruner: PrunerConfig | None = Field(
        default_factory=MedianConfig,
        description="Stops weak trials early on training.monitor. None never prunes.",
    )
    params: list[ParamConfig] = Field(
        min_length=1,
        description="The search space. The data and the objective are fixed: "
        "data.datasets, data.component_order, data.sampling_rate, "
        "augmentation.label_events and training.monitor cannot be tuned.",
    )

    @model_validator(mode="after")
    def _check_paths(self) -> Self:
        paths = [param.path for param in self.params]
        if len(set(paths)) != len(paths):
            raise ValueError("each parameter path may appear only once")
        for path in paths:
            if any(path == f or path.startswith(f"{f}.") for f in FIXED_PATHS):
                raise ValueError(f"{path!r} cannot be tuned (fixed: {FIXED_PATHS})")
        return self


def set_path(data: dict[str, JsonValue], path: str, value: JsonValue) -> None:
    """Set a dotted path in a nested config dict; list items by index."""
    *parents, last = path.split(".")
    node: JsonValue = data
    for key in parents:
        node = _child(node, key, path)
    _child(node, last, path)  # the target must exist
    if isinstance(node, list):
        node[int(last)] = value
    elif isinstance(node, dict):
        node[last] = value


def _child(node: JsonValue, key: str, path: str) -> JsonValue:
    if isinstance(node, list) and key.isdigit() and int(key) < len(node):
        return node[int(key)]
    if isinstance(node, dict) and key in node:
        return node[key]
    raise ValueError(f"unknown config path {path!r}")
