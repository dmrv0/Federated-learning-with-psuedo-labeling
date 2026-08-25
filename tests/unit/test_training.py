import numpy as np
import pytest
import torch

from ssfl.config import Backbone
from ssfl.data.datasets import TensorFeatureDataset
from ssfl.models import build_classifier
from ssfl.seeding import seed_everything
from ssfl.training import evaluate, make_loader, predict_probs, train_distillation, train_supervised


def _synthetic_classification_dataset(seed: int = 0, n: int = 40) -> TensorFeatureDataset:
    rng = np.random.default_rng(seed)
    flat = rng.random((n, 115), dtype=np.float64).astype(np.float32)
    reshaped = flat.reshape(n, 5, 23).transpose(0, 2, 1)  # any (n,23,5)-shaped array is fine here
    labels = rng.integers(0, 11, size=n).astype(np.int64)
    return TensorFeatureDataset(flat, reshaped, Backbone.cnn, labels)


def _available_devices() -> list[torch.device]:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    return devices


def test_train_supervised_deterministic_regression() -> None:
    def run_once() -> float:
        seed_everything(2023)
        model = build_classifier(Backbone.cnn)
        dataset = _synthetic_classification_dataset()
        loader = make_loader(dataset, batch_size=8, shuffle=True, seed=2023)
        result = train_supervised(model, loader, torch.device("cpu"), epochs=2, lr=1e-3)
        return result.final_loss

    loss_a = run_once()
    loss_b = run_once()
    assert loss_a == pytest.approx(loss_b, rel=1e-6)


@pytest.mark.parametrize("device", _available_devices())
def test_train_supervised_valid_outputs_on_device(device: torch.device) -> None:
    seed_everything(2023)
    model = build_classifier(Backbone.cnn)
    dataset = _synthetic_classification_dataset()
    loader = make_loader(dataset, batch_size=8, shuffle=True, seed=2023)
    result = train_supervised(model, loader, device, epochs=1, lr=1e-3)
    assert len(result.epoch_losses) == 1
    assert torch.isfinite(torch.tensor(result.final_loss))


def test_train_distillation_soft_targets() -> None:
    seed_everything(2023)
    model = build_classifier(Backbone.cnn)
    rng = np.random.default_rng(1)
    x = torch.from_numpy(rng.random((16, 23, 5), dtype=np.float64).astype(np.float32))
    raw = rng.random((16, 11))
    soft = torch.from_numpy((raw / raw.sum(axis=1, keepdims=True)).astype(np.float32))
    loader = [(x[i : i + 8], soft[i : i + 8]) for i in range(0, 16, 8)]
    result = train_distillation(model, loader, torch.device("cpu"), epochs=1, lr=1e-3)
    assert torch.isfinite(torch.tensor(result.final_loss))


def test_evaluate_accuracy_in_range() -> None:
    model = build_classifier(Backbone.cnn)
    dataset = _synthetic_classification_dataset()
    loader = make_loader(dataset, batch_size=8, shuffle=False, seed=2023)
    metrics = evaluate(model, loader, torch.device("cpu"))
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["loss"] >= 0.0


def test_predict_probs_shape_order_and_normalization() -> None:
    model = build_classifier(Backbone.cnn)
    n = 20
    rng = np.random.default_rng(2)
    flat = rng.random((n, 115), dtype=np.float64).astype(np.float32)
    reshaped = flat.reshape(n, 5, 23).transpose(0, 2, 1)
    dataset = TensorFeatureDataset(flat, reshaped, Backbone.cnn, labels=None)
    loader = make_loader(dataset, batch_size=7, shuffle=False, seed=2023)
    probs = predict_probs(model, loader, torch.device("cpu"))
    assert probs.shape == (n, 11)
    assert torch.allclose(probs.sum(dim=1), torch.ones(n), atol=1e-5)
