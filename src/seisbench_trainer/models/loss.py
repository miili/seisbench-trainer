"""Training losses. Each loss is one member of a model's ``loss`` union.

A loss config builds a ``LossFunction`` for one model: a callable on the raw
model output (logits) and the batch. torch is imported only when a loss is
built, so that config handling stays fast.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Protocol

from pydantic import Field, NonNegativeFloat, PositiveFloat

from seisbench_trainer.schema import StrictModel

if TYPE_CHECKING:
    from torch import Tensor

    from seisbench_trainer.data.loaders import Batch


class LossFunction(Protocol):
    def __call__(self, logits: Tensor, batch: Batch) -> Tensor: ...


def _noise_down_weighted() -> dict[str, float]:
    return {"N": 0.5}


class CrossEntropyLoss(StrictModel):
    """Soft-target cross-entropy over the softmax output channels (PhaseNet).

    Per sample, ``-sum_c w_c * y_c * log(q_c)``; averaged over the samples
    and the batch, leaving out the samples where the batch's ``label_mask``
    (``augmentation.missing_picks``) marks an unread phase. The default takes
    ``log_softmax`` of the logits, as the original PhaseNet, and weighs the
    noise channel 0.5. ``from_logits=False``
    and ``weights={}`` give SeisBench's pick-benchmark ``vector_cross_entropy``
    (``log(softmax + eps)``, unweighted).
    """

    type: Literal["cross_entropy"] = "cross_entropy"
    weights: dict[str, NonNegativeFloat] = Field(
        default_factory=_noise_down_weighted,
        description="Weight per output channel, by label; missing channels weigh "
        "1. The default N 0.5 balances P and S against the far more frequent "
        "noise samples: on BohemiaSaxony it gave the best F1 and pick times.",
    )
    from_logits: bool = Field(
        default=True,
        description="True takes log_softmax of the logits (numerically stable, as "
        "in the original PhaseNet). False takes log(softmax + eps), as in "
        "SeisBench's pick-benchmark.",
    )
    eps: PositiveFloat = Field(
        default=1e-5, description="Added to the softmax when from_logits is false."
    )

    def build(self, labels: str) -> LossFunction:
        """The loss for a model with these output ``labels`` (e.g. 'PSN')."""
        unknown = set(self.weights) - set(labels)
        if unknown:
            raise ValueError(
                f"loss weights for unknown labels {sorted(unknown)}; "
                f"the model outputs {labels!r}"
            )
        return _CrossEntropy(
            [self.weights.get(label, 1.0) for label in labels],
            self.from_logits,
            self.eps,
        )


class _CrossEntropy:
    def __init__(self, weights: list[float], from_logits: bool, eps: float) -> None:
        self.weights = weights
        self.from_logits = from_logits
        self.eps = eps

    def __call__(self, logits: Tensor, batch: Batch) -> Tensor:
        import torch

        if self.from_logits:
            log_q = torch.log_softmax(logits, dim=1)
        else:
            log_q = torch.log(torch.softmax(logits, dim=1) + self.eps)
        weights = torch.tensor(self.weights, dtype=log_q.dtype, device=log_q.device)
        target = batch["y"] * weights[None, :, None]
        per_sample = -(target * log_q).sum(dim=1)
        if "label_mask" not in batch:
            return per_sample.mean()
        # samples where a phase may arrive unread leave the loss (all channels)
        mask = batch["label_mask"].amin(dim=1).to(per_sample.dtype)
        return (per_sample * mask).sum() / mask.sum().clamp(min=1.0)


PhaseNetLoss = Annotated[CrossEntropyLoss, Field(discriminator="type")]
