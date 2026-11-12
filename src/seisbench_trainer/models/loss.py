"""Training losses. Each loss is one member of a model's ``loss`` union.

A loss config builds a ``LossFunction`` for one model: a callable on the raw
model output (logits) and the batch. torch is imported only when a loss is
built, so that config handling stays fast.
"""

from typing import TYPE_CHECKING, Annotated, Literal, Protocol

from pydantic import Field, NonNegativeFloat, PositiveFloat, field_validator

from seisbench_trainer.schema import StrictModel
from seisbench_trainer.units import seconds_to_samples

if TYPE_CHECKING:
    from torch import Tensor

    from seisbench_trainer.data.loaders import Batch


class LossFunction(Protocol):
    def __call__(self, logits: "Tensor", batch: "Batch") -> "Tensor": ...


class CrossEntropyLoss(StrictModel):
    """Soft-target cross-entropy over the softmax output channels (PhaseNet).

    Per sample, ``-sum_c w_c * y_c * log(q_c)``; averaged over the samples of
    each window, then over the batch. With the defaults this is SeisBench's
    pick-benchmark ``vector_cross_entropy``: ``log(softmax + eps)``, no
    weights, no mask.
    """

    type: Literal["cross_entropy"] = "cross_entropy"
    weights: dict[str, NonNegativeFloat] = Field(
        default_factory=dict[str, NonNegativeFloat],
        description="Weight per output channel, by label (e.g. {'P': 1, 'S': 1, "
        "'N': 0.5}). Missing channels weigh 1.",
    )
    from_logits: bool = Field(
        default=False,
        description="True takes log_softmax of the logits (numerically stable, as "
        "in the original PhaseNet). False takes log(softmax + eps), as in "
        "SeisBench's pick-benchmark.",
    )
    eps: PositiveFloat = Field(
        default=1e-5, description="Added to the softmax when from_logits is false."
    )
    pick_mask_s: Literal[False] | PositiveFloat = Field(
        default=False,
        description="In windows with picks, only samples within this distance of "
        "a pick enter the loss; the noise far from the picks is masked out. "
        "Windows without picks are never masked. Must cover the label curve "
        "(at least 3 label sigmas). false (or null) disables the mask.",
    )

    @field_validator("pick_mask_s", mode="before")
    @classmethod
    def _mask_off(cls, value: object) -> object:
        if value is None:
            return False
        if value is True:
            raise ValueError("pick_mask_s takes a distance in seconds or false")
        return value

    def build(self, labels: str, sampling_rate: float) -> LossFunction:
        """The loss for a model with these output ``labels`` (e.g. 'PSN')."""
        unknown = set(self.weights) - set(labels)
        if unknown:
            raise ValueError(
                f"loss weights for unknown labels {sorted(unknown)}; "
                f"the model outputs {labels!r}"
            )
        mask_samples = (
            None
            if self.pick_mask_s is False
            else seconds_to_samples(self.pick_mask_s, sampling_rate)
        )
        return _CrossEntropy(
            [self.weights.get(label, 1.0) for label in labels],
            self.from_logits,
            self.eps,
            mask_samples,
        )


class _CrossEntropy:
    def __init__(
        self,
        weights: list[float],
        from_logits: bool,
        eps: float,
        mask_samples: int | None,
    ) -> None:
        self.weights = weights
        self.from_logits = from_logits
        self.eps = eps
        self.mask_samples = mask_samples

    def __call__(self, logits: "Tensor", batch: "Batch") -> "Tensor":
        import torch

        if self.from_logits:
            log_q = torch.log_softmax(logits, dim=1)
        else:
            log_q = torch.log(torch.softmax(logits, dim=1) + self.eps)
        weights = torch.tensor(self.weights, dtype=log_q.dtype, device=log_q.device)
        target = batch["y"] * weights[None, :, None]
        per_sample = -(target * log_q).sum(dim=1)  # (batch, sample)
        if self.mask_samples is None:
            return per_sample.mean()
        mask = pick_mask(batch["pick_samples"], per_sample.shape[-1], self.mask_samples)
        per_window = (per_sample * mask).sum(dim=-1) / mask.sum(dim=-1)
        return per_window.mean()


def pick_mask(pick_samples: "Tensor", n_samples: int, half_width: int) -> "Tensor":
    """Samples within ``half_width`` of a pick; all samples if a window has none.

    ``pick_samples`` has shape (batch, phase, max_picks), NaN-padded (from
    ``data.augmentations.PickSamples``). Returns a float mask (batch, sample).
    """
    import torch

    picks = pick_samples.flatten(1)  # (batch, picks)
    times = torch.arange(n_samples, device=picks.device, dtype=picks.dtype)
    near = (times[None, None, :] - picks[:, :, None]).abs() <= half_width
    has_pick = (~picks.isnan()).any(dim=1, keepdim=True)
    return (near.any(dim=1) | ~has_pick).to(picks.dtype)


PhaseNetLoss = Annotated[CrossEntropyLoss, Field(discriminator="type")]
