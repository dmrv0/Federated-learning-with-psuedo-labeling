"""DS-FL baseline: full open-data soft-prediction averaging with temperature sharpening."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import TensorDataset

from ssfl.models import SSFLModel
from ssfl.training import TrainResult, evaluate, make_loader, predict_probs, train_distillation
from ssfl.telemetry import EventCallback


@dataclass(frozen=True)
class SoftPredictionUpload:
    client_id: str
    probs: np.ndarray  # (num_open, num_classes) float32


def client_predict_step(
    client_id: str,
    classifier: SSFLModel,
    open_dataset,
    device: torch.device,
    batch_size: int,
    seed: int,
    event_callback: EventCallback | None = None,
) -> SoftPredictionUpload:
    loader = make_loader(open_dataset, batch_size, shuffle=False, seed=seed)
    probs = (
        predict_probs(
            classifier,
            loader,
            device,
            event_callback=event_callback,
            stage="client_dsfl_open_prediction",
        )
        .numpy()
        .astype(np.float32)
    )
    return SoftPredictionUpload(client_id, probs)


def aggregate_mean(uploads: list[SoftPredictionUpload]) -> np.ndarray:
    """Arithmetic mean across (deduped-by-client) uploads -- idempotent against a duplicated
    submission from the same client.

    Stacked in ``client_id``-sorted order, not upload/arrival order: ``.mean(axis=0)`` sums in
    array order, floating-point addition isn't associative, and Ray/Flower reply arrival order
    isn't guaranteed reproducible across runs of the same seeded simulation -- arrival-order
    stacking made this mean (and, after ``sharpen``'s ``1/T=10`` power, the resulting hard
    training targets) differ between two otherwise-identical runs. Found via a failing
    ``test_identical_seed_runs_are_deterministic``."""
    seen: set[str] = set()
    deduped: list[SoftPredictionUpload] = []
    for u in sorted(uploads, key=lambda u: u.client_id):
        if u.client_id in seen:
            continue
        seen.add(u.client_id)
        deduped.append(u)
    if not deduped:
        raise ValueError("aggregate_mean: no uploads")
    return np.stack([u.probs for u in deduped], axis=0).mean(axis=0)


def sharpen(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Softmax temperature sharpening: ``p_i^(1/T) / sum_j p_j^(1/T)``. Paper default ``T=0.1``
    (sharpens); ``T=1`` is a no-op."""
    powered = np.power(np.clip(probs, 1e-12, None), 1.0 / temperature)
    return powered / powered.sum(axis=1, keepdims=True)


def distill_step(
    model: SSFLModel,
    open_dataset,
    sharpened_targets: np.ndarray,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    event_callback: EventCallback | None = None,
) -> TrainResult:
    """Client- or server-side distillation on the sharpened global soft targets -- same function,
    since both sides train identically on the same (open_x, sharpened_targets) pairs."""
    targets = torch.from_numpy(sharpened_targets).float()
    loader = make_loader(
        TensorDataset(open_dataset.x, targets), batch_size, shuffle=True, seed=seed
    )
    return train_distillation(
        model,
        loader,
        device,
        epochs,
        lr,
        event_callback=event_callback,
        stage="dsfl_distillation",
    )


def server_evaluate(
    model: SSFLModel,
    test_dataset,
    device: torch.device,
    batch_size: int,
    seed: int,
    event_callback: EventCallback | None = None,
) -> dict[str, float]:
    loader = make_loader(test_dataset, batch_size, shuffle=False, seed=seed)
    return evaluate(
        model,
        loader,
        device,
        event_callback=event_callback,
        stage="server_dsfl_test",
    )
