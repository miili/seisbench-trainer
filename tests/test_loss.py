from __future__ import annotations

import pytest
import torch

from seisbench_trainer.data.loaders import Batch
from seisbench_trainer.models.loss import CrossEntropyLoss

N_SAMPLES = 400


def make_batch(n_windows: int) -> Batch:
    """Random soft targets in 'PSN' order."""
    y = torch.rand(n_windows, 3, N_SAMPLES)
    return {"y": y / y.sum(dim=1, keepdim=True)}


def test_eps_variant_is_the_reference_loss() -> None:
    batch = make_batch(2)
    logits = torch.randn(2, 3, N_SAMPLES)
    reference = batch["y"] * torch.log(logits.softmax(dim=1) + 1e-5)
    expected = -reference.mean(dim=-1).sum(dim=-1).mean()
    loss = CrossEntropyLoss(from_logits=False, weights={}).build("PSN")
    torch.testing.assert_close(loss(logits, batch), expected)


def test_from_logits_uses_log_softmax() -> None:
    batch = make_batch(1)
    logits = 30 * torch.randn(1, 3, N_SAMPLES)  # saturated softmax
    expected = -(batch["y"] * logits.log_softmax(dim=1)).sum(dim=1).mean()
    loss = CrossEntropyLoss(weights={}).build("PSN")
    torch.testing.assert_close(loss(logits, batch), expected)
    reference = CrossEntropyLoss(from_logits=False, weights={}).build("PSN")
    assert reference(logits, batch) < loss(logits, batch)  # eps caps the loss


def test_default_weighs_noise_half() -> None:
    batch = make_batch(1)
    logits = torch.randn(1, 3, N_SAMPLES)
    terms = -(batch["y"] * logits.log_softmax(dim=1)).mean(dim=-1)[0]
    loss = CrossEntropyLoss().build("PSN")
    expected = terms[0] + terms[1] + 0.5 * terms[2]
    torch.testing.assert_close(loss(logits, batch), expected)


def test_weights_follow_labels() -> None:
    batch = make_batch(1)
    logits = torch.randn(1, 3, N_SAMPLES)
    config = CrossEntropyLoss(weights={"N": 0.0, "S": 2.0})
    psn = config.build("PSN")(logits, batch)
    order = [2, 0, 1]  # the same channels in 'NPS' order
    nps = config.build("NPS")(logits[:, order], {"y": batch["y"][:, order]})
    torch.testing.assert_close(psn, nps)
    terms = -(batch["y"] * logits.log_softmax(dim=1)).mean(dim=-1)[0]
    torch.testing.assert_close(psn, terms[0] + 2 * terms[1])


def test_unknown_weight_label() -> None:
    with pytest.raises(ValueError, match=r"unknown labels \['D'\]"):
        CrossEntropyLoss(weights={"D": 1.0}).build("PSN")
