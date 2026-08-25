import math

import numpy as np
import pytest
import torch

from ssfl.config import Backbone
from ssfl.data.datasets import TensorFeatureDataset
from ssfl.models import build_classifier
from ssfl.protocols.fl import ClientUpdate, TrainResult, federated_average, server_evaluate

DEVICE = torch.device("cpu")


def _dataset(n: int, seed: int) -> TensorFeatureDataset:
    rng = np.random.default_rng(seed)
    flat = rng.random((n, 115), dtype=np.float64).astype(np.float32)
    reshaped = flat.reshape(n, 5, 23).transpose(0, 2, 1)
    labels = rng.integers(0, 11, size=n).astype(np.int64)
    return TensorFeatureDataset(flat, reshaped, Backbone.cnn, labels)


def _update(client_id: str, w: list[float], n: int) -> ClientUpdate:
    return ClientUpdate(client_id, {"w": torch.tensor(w)}, n, TrainResult())


def test_federated_average_is_sample_weighted() -> None:
    updates = [_update("a", [1.0, 2.0], 10), _update("b", [3.0, 4.0], 30)]
    averaged = federated_average(updates)
    assert torch.allclose(averaged["w"], torch.tensor([2.5, 3.5]))


def test_federated_average_idempotent_under_duplicate_client() -> None:
    once = federated_average([_update("a", [1.0], 10), _update("b", [3.0], 30)])
    duplicated = federated_average(
        [_update("a", [1.0], 10), _update("a", [999.0], 10), _update("b", [3.0], 30)]
    )
    assert torch.allclose(once["w"], duplicated["w"])


def test_federated_average_rejects_empty() -> None:
    with pytest.raises(ValueError):
        federated_average([])


def test_federated_average_rejects_zero_total_examples() -> None:
    with pytest.raises(ValueError):
        federated_average([_update("a", [1.0], 0), _update("b", [2.0], 0)])


def test_server_evaluate() -> None:
    model = build_classifier(Backbone.cnn)
    global_state = model.state_dict()
    test_ds = _dataset(20, seed=1)
    metrics = server_evaluate(model, global_state, test_ds, DEVICE, batch_size=8, seed=0)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert math.isfinite(metrics["loss"])
