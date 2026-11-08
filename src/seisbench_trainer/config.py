"""Top-level run config. One JSON file describes one run."""

from pathlib import Path
from typing import Literal

from pydantic import (
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)

from seisbench_trainer.data import AugmentationConfig, DataConfig
from seisbench_trainer.models import ModelConfig, PhaseNetConfig
from seisbench_trainer.schema import StrictModel
from seisbench_trainer.training.optim import (
    AdamConfig,
    OptimizerConfig,
    SchedulerConfig,
)


class TrainingConfig(StrictModel):
    epochs: PositiveInt = 50
    seed: int = 42
    accelerator: Literal["auto", "cpu", "gpu"] = "auto"
    precision: Literal["32-true", "16-mixed", "bf16-mixed"] = "32-true"
    selection_metric: str = Field(
        default="val_loss",
        description="Metric for checkpointing, early stopping, HPO and the registry.",
    )
    selection_mode: Literal["min", "max"] = "min"
    early_stopping_patience: PositiveInt | None = Field(
        default=None, description="Epochs without improvement. None disables it."
    )
    optimizer: OptimizerConfig = Field(default_factory=AdamConfig)
    scheduler: SchedulerConfig | None = None


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
        default_factory=lambda: [0.1, 0.2, 0.5], min_length=1
    )
    min_peak_height: float = Field(default=0.05, gt=0.0, lt=1.0)
    min_peak_distance_s: PositiveFloat = 1.0
    edge_mask_s: NonNegativeFloat = Field(
        default=1.0,
        description="Samples this close to the trace edges are ignored for "
        "noise scores.",
    )
    baseline: str | None = Field(
        default="original",
        description="SeisBench pretrained weights evaluated alongside for "
        "comparison. None disables the baseline.",
    )
    plot_every_n_epochs: NonNegativeInt = Field(
        default=1, description="0 disables the figures."
    )
    test_at_end: bool = False


class TrackingConfig(StrictModel):
    experiment: str | None = Field(
        default=None,
        description="MLflow experiment. None derives it from datasets and model.",
    )
    run_name: str | None = None
    tags: dict[str, str] = Field(default_factory=dict[str, str])


class TrainConfig(StrictModel):
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=PhaseNetConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)


def load_config(path: Path) -> TrainConfig:
    """Read and validate a JSON run config."""
    return TrainConfig.model_validate_json(path.read_text())


def dump_config(config: TrainConfig) -> str:
    """Serialise a run config to indented JSON."""
    return config.model_dump_json(indent=2)
