import math

import pytest
import torch

from seisbench_trainer.data.loaders import Batch
from seisbench_trainer.models.loss import CrossEntropyLoss, pick_mask

RATE = 100.0
N_SAMPLES = 400


def make_batch(picks: list[list[float]]) -> Batch:
    """Soft targets in 'PSN' order and pick_samples, one window per entry."""
    y = torch.rand(len(picks), 3, N_SAMPLES)
    y = y / y.sum(dim=1, keepdim=True)
    pick_samples = torch.full((len(picks), 2, 4), math.nan)
    for row, window in enumerate(picks):
        pick_samples[row, 0, : len(window)] = torch.tensor(window)
    return {"y": y, "pick_samples": pick_samples}


def test_default_is_the_reference_loss() -> None:
    batch = make_batch([[], []])
    logits = torch.randn(2, 3, N_SAMPLES)
    reference = batch["y"] * torch.log(logits.softmax(dim=1) + 1e-5)
    expected = -reference.mean(dim=-1).sum(dim=-1).mean()
    loss = CrossEntropyLoss().build("PSN", RATE)
    torch.testing.assert_close(loss(logits, batch), expected)


def test_from_logits_uses_log_softmax() -> None:
    batch = make_batch([[]])
    logits = 30 * torch.randn(1, 3, N_SAMPLES)  # saturated softmax
    expected = -(batch["y"] * logits.log_softmax(dim=1)).sum(dim=1).mean()
    loss = CrossEntropyLoss(from_logits=True).build("PSN", RATE)
    torch.testing.assert_close(loss(logits, batch), expected)
    reference = CrossEntropyLoss().build("PSN", RATE)
    assert reference(logits, batch) < loss(logits, batch)  # eps caps the loss


def test_weights_follow_labels() -> None:
    batch = make_batch([[]])
    logits = torch.randn(1, 3, N_SAMPLES)
    config = CrossEntropyLoss(weights={"N": 0.0, "S": 2.0})
    psn = config.build("PSN", RATE)(logits, batch)
    order = [2, 0, 1]  # the same channels in 'NPS' order
    nps = config.build("NPS", RATE)(
        logits[:, order], {**batch, "y": batch["y"][:, order]}
    )
    torch.testing.assert_close(psn, nps)
    log_q = torch.log(logits.softmax(dim=1) + 1e-5)
    terms = -(batch["y"] * log_q).mean(dim=-1)[0]
    torch.testing.assert_close(psn, terms[0] + 2 * terms[1])


def test_unknown_weight_label() -> None:
    with pytest.raises(ValueError, match=r"unknown labels \['D'\]"):
        CrossEntropyLoss(weights={"D": 1.0}).build("PSN", RATE)


def test_pick_mask() -> None:
    picks = torch.full((2, 2, 4), math.nan)
    picks[0, 0, 0] = 100.0
    picks[0, 1, 0] = 300.0
    mask = pick_mask(picks, N_SAMPLES, half_width=10)
    expected = torch.zeros(N_SAMPLES)
    expected[90:111] = 1
    expected[290:311] = 1
    torch.testing.assert_close(mask[0], expected)
    torch.testing.assert_close(mask[1], torch.ones(N_SAMPLES))  # no picks


def test_mask_ignores_noise_far_from_picks() -> None:
    batch = make_batch([[200.0], []])
    logits = torch.randn(2, 3, N_SAMPLES)
    changed = logits.clone()
    changed[:, 0, :50] += 5.0  # far from the pick at 200
    loss = CrossEntropyLoss(pick_mask_s=0.5).build("PSN", RATE)
    torch.testing.assert_close(
        loss(logits[:1], {k: v[:1] for k, v in batch.items()}),
        loss(changed[:1], {k: v[:1] for k, v in batch.items()}),
    )
    pick_free = {k: v[1:] for k, v in batch.items()}
    assert loss(logits[1:], pick_free) != loss(changed[1:], pick_free)
    unmasked = CrossEntropyLoss().build("PSN", RATE)
    torch.testing.assert_close(
        loss(logits[1:], pick_free), unmasked(logits[1:], pick_free)
    )
