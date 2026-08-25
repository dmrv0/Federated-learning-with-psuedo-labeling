"""FL baseline: sample-weighted FedAvg. Pure functions over ``state_dict``s -- no Flower."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ssfl.models import SSFLModel
from ssfl.training import TrainResult, evaluate, make_loader, train_supervised
from ssfl.telemetry import EventCallback


@dataclass
class ClientUpdate:
    client_id: str
    state_dict: dict[str, torch.Tensor]
    num_examples: int
    train_result: TrainResult


def client_train_step(
    client_id: str,
    model: SSFLModel,
    private_dataset,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    event_callback: EventCallback | None = None,
) -> ClientUpdate:
    """Local training starting from the current global weights already loaded into ``model``.
    Returns the full updated ``state_dict`` -- FL, unlike SSFL, is defined by uploading model
    parameters."""
    loader = make_loader(private_dataset, batch_size, shuffle=True, seed=seed)
    result = train_supervised(
        model,
        loader,
        device,
        epochs,
        lr,
        event_callback=event_callback,
        stage="client_fl_supervised",
    )
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return ClientUpdate(client_id, state, len(private_dataset), result)


def federated_average(updates: list[ClientUpdate]) -> dict[str, torch.Tensor]:
    """Sample-weighted average of client ``state_dict``s. Idempotent: a duplicate ``client_id``
    (e.g. a retried upload) is deduped, first submission kept, before weighting."""
    seen: set[str] = set()
    deduped: list[ClientUpdate] = []
    for u in updates:
        if u.client_id in seen:
            continue
        seen.add(u.client_id)
        deduped.append(u)
    if not deduped:
        raise ValueError("federated_average: no updates")

    total = sum(u.num_examples for u in deduped)
    if total == 0:
        raise ValueError("federated_average: updates carry zero total examples")

    averaged: dict[str, torch.Tensor] = {}
    for key, ref in deduped[0].state_dict.items():
        stacked = torch.stack(
            [u.state_dict[key].float() * (u.num_examples / total) for u in deduped]
        )
        averaged[key] = stacked.sum(dim=0).to(ref.dtype)
    return averaged


def server_evaluate(
    model: SSFLModel,
    global_state: dict[str, torch.Tensor],
    test_dataset,
    device: torch.device,
    batch_size: int,
    seed: int,
    event_callback: EventCallback | None = None,
) -> dict[str, float]:
    model.load_state_dict(global_state)
    loader = make_loader(test_dataset, batch_size, shuffle=False, seed=seed)
    return evaluate(model, loader, device, event_callback=event_callback, stage="server_fl_test")
