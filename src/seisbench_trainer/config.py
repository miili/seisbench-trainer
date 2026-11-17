"""Top-level run config. One JSON file describes one run."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import (
    Field,
    JsonValue,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    ValidationError,
    model_validator,
)

from seisbench_trainer.data import AugmentationConfig, DataConfig
from seisbench_trainer.data.steps import NoisyValidation
from seisbench_trainer.hpo.config import HPOConfig, set_path
from seisbench_trainer.models import ModelConfig, PhaseNetConfig
from seisbench_trainer.schema import StrictModel, UserPath
from seisbench_trainer.training.optim import (
    AdamConfig,
    CosineConfig,
    Mode,
    OptimizerConfig,
    PlateauConfig,
    SchedulerConfig,
)

Monitor = Literal["val_loss", "det/f1_mean", "det_P/f1", "det_S/f1"]
"""Validation metrics that can rank models. ``val_loss`` is minimized, the F1
scores are maximized."""


class TrainingConfig(StrictModel):
    """Training loop. By default every run finds its own length: the plateau
    scheduler halves the learning rate after 5 epochs without improvement and
    early stopping ends the run after 15; ``epochs`` is only the limit."""

    epochs: PositiveInt = Field(
        default=100,
        description="Upper limit; early stopping usually ends the run sooner.",
    )
    seed: int = Field(
        default=42,
        description="Seeds weights, the training augmentations and the fixed "
        "validation and test windows. Runs with the same config and seed are "
        "reproducible; compare settings with two seeds (run-to-run noise of "
        "det/f1_mean is about 0.002).",
    )
    accelerator: Literal["auto", "cpu", "gpu"] = Field(
        default="auto",
        description="Lightning accelerator; a run always uses one device. "
        "'auto' takes the GPU if there is one.",
    )
    precision: Literal["32-true", "16-mixed", "bf16-mixed"] = Field(
        default="32-true",
        description="Lightning precision. Mixed precision is faster on GPUs but "
        "has not been compared against 32-true.",
    )
    monitor: Monitor = Field(
        default="det/f1_mean",
        description="Validation metric that picks the best model: checkpointing, "
        "early stopping, the plateau scheduler, HPO and the registry. val_loss "
        "is minimized, the F1 scores are maximized.",
    )
    early_stopping_patience: PositiveInt | None = Field(
        default=15,
        description="Stop after this many epochs without improvement. None "
        "disables it (needed for the cosine scheduler).",
    )
    min_delta: NonNegativeFloat = Field(
        default=0.002,
        description="Smallest change of monitor that counts as an improvement, "
        "for early stopping and the plateau scheduler. 0.002 is the run-to-run "
        "noise of det/f1_mean on BohemiaSaxony; use about 0.0005 for val_loss.",
    )
    optimizer: OptimizerConfig = Field(
        default_factory=AdamConfig,
        description="Optimizer and its learning rate ('adam' or 'adamw').",
    )
    scheduler: SchedulerConfig | None = Field(
        default_factory=PlateauConfig,
        description="Learning-rate schedule. None keeps the rate constant.",
    )
    checkpoint_dir: UserPath = Field(
        default=Path("checkpoints"),
        description="Local or scratch directory for Lightning checkpoints. "
        "They are not uploaded to MLflow.",
    )

    @model_validator(mode="after")
    def _cosine_runs_to_the_end(self) -> Self:
        if (
            isinstance(self.scheduler, CosineConfig)
            and self.early_stopping_patience is not None
        ):
            raise ValueError(
                "the cosine scheduler needs a fixed length: set "
                "training.early_stopping_patience to null"
            )
        return self

    @property
    def monitor_mode(self) -> Mode:
        """Direction of ``monitor``: 'min' for the loss, 'max' for F1."""
        return "min" if self.monitor == "val_loss" else "max"


class EvaluationConfig(StrictModel):
    """Validation and test evaluation. Durations in seconds."""

    pick_window_s: PositiveFloat = Field(
        default=2.0,
        description="Half-width of the window around the expected pick for the "
        "pick-accuracy metrics.",
    )
    tolerance_s: PositiveFloat = Field(
        default=0.5, description="Matching tolerance of the detection metrics."
    )
    test_tolerances_s: list[PositiveFloat] = Field(
        default_factory=lambda: [0.1, 0.2, 0.5],
        min_length=1,
        description="Matching tolerances of the test evaluation (one set of "
        "scores each) and of the 'within_<t>s' pick-accuracy fractions.",
    )
    min_peak_height: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Lowest probability peak that counts as a detection; the "
        "threshold curves start above it.",
    )
    min_peak_distance_s: PositiveFloat = Field(
        default=1.0,
        description="Closer peaks of one phase are merged into the higher one.",
    )
    edge_mask_s: NonNegativeFloat = Field(
        default=1.0,
        description="Samples this close to the trace edges are ignored for "
        "noise scores.",
    )
    baseline: ModelConfig | None = Field(
        default_factory=lambda: PhaseNetConfig(pretrained="original"),
        description="Pretrained model evaluated alongside for comparison, fed "
        "in its own component order. None disables the baseline.",
    )
    noisy_validation: NoisyValidation | None = Field(
        default_factory=NoisyValidation,
        description="Also validate on the dev windows with real noise from the "
        "dev split's noise traces added (metrics 'noisy/...'). Skipped if the "
        "dev split has no noise traces. None disables it.",
    )
    plot_every_n_epochs: NonNegativeInt = Field(
        default=1, description="0 disables the figures."
    )
    test_at_end: bool = Field(
        default=False,
        description="Evaluate the best model and the baseline on the test split "
        "after training (thresholds from the dev split). HPO trials never do.",
    )


class TrackingConfig(StrictModel):
    """Where the run is logged. The tracking server or folder comes from the
    MLFLOW_TRACKING_URI environment variable, not from the config."""

    experiment: str | None = Field(
        default=None,
        description="MLflow experiment. None derives it from datasets and model.",
    )
    run_name: str | None = Field(
        default=None, description="MLflow run name. None lets MLflow pick one."
    )
    tags: dict[str, str] = Field(
        default_factory=dict[str, str],
        description='Extra MLflow run tags, e.g. {"purpose": "lr-grid"}.',
    )


class TrainConfig(StrictModel):
    """One training run. Every field has a default, so ``{}`` is a valid
    config; ``seisbench-trainer config`` prints them all."""

    data: DataConfig = Field(
        default_factory=DataConfig,
        description="Datasets, component order, sampling rate and loaders.",
    )
    model: ModelConfig = Field(
        default_factory=PhaseNetConfig,
        description="Picker model, its starting weights and its loss.",
    )
    augmentation: AugmentationConfig = Field(
        default_factory=AugmentationConfig,
        description="Training windows and perturbations, normalization and labels.",
    )
    training: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description="Training loop: length, seed, optimizer, scheduler, model "
        "selection.",
    )
    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig,
        description="Validation and test metrics, baseline model, figures.",
    )
    tracking: TrackingConfig = Field(
        default_factory=TrackingConfig,
        description="MLflow experiment, run name and tags.",
    )
    hpo: HPOConfig | None = Field(
        default=None, description="Hyperparameter search; used by the hpo command."
    )

    @model_validator(mode="after")
    def _models_run_at_data_rate(self) -> Self:
        rate = self.data.sampling_rate
        for name, model in (
            ("model", self.model),
            ("evaluation.baseline", self.evaluation.baseline),
        ):
            if model is not None and model.sampling_rate not in (None, rate):
                raise ValueError(
                    f"{name}.sampling_rate ({model.sampling_rate:g} Hz) must "
                    f"equal data.sampling_rate ({rate:g} Hz)"
                )
        return self

    @model_validator(mode="after")
    def _check_search_space(self) -> Self:
        if self.hpo is not None:
            try:
                self.with_values({p.path: p.example() for p in self.hpo.params})
            except ValidationError as error:
                raise ValueError(
                    f"hpo params give an invalid config: {error}"
                ) from error
        return self

    def with_values(self, values: dict[str, JsonValue]) -> TrainConfig:
        """A validated copy with dotted-path values applied and no hpo section."""
        data: dict[str, JsonValue] = self.model_dump(mode="json")
        for path, value in values.items():
            set_path(data, path, value)
        data["hpo"] = None
        return TrainConfig.model_validate(data)


def load_config(path: Path) -> TrainConfig:
    """Read and validate a JSON run config."""
    return TrainConfig.model_validate_json(path.read_text())


def dump_config(config: TrainConfig) -> str:
    """Serialise a run config to indented JSON."""
    return config.model_dump_json(indent=2)
