import math

import numpy as np
import torch

from ssfl.config import Backbone
from ssfl.data.datasets import TensorFeatureDataset
from ssfl.models import NUM_CLASSES, build_classifier
from ssfl.protocols.dsfl import SoftPredictionUpload, aggregate_mean, distill_step, server_evaluate, sharpen

DEVICE = torch.device("cpu")


def _dataset(n: int, seed: int, labeled: bool) -> TensorFeatureDataset:
    rng = np.random.default_rng(seed)
    flat = rng.random((n, 115), dtype=np.float64).astype(np.float32)
    reshaped = flat.reshape(n, 5, 23).transpose(0, 2, 1)
    labels = rng.integers(0, 11, size=n).astype(np.int64) if labeled else None
    return TensorFeatureDataset(flat, reshaped, Backbone.cnn, labels)


def test_aggregate_mean() -> None:
    upload_a = SoftPredictionUpload("a", np.array([[0.2, 0.8], [0.5, 0.5]], dtype=np.float32))
    upload_b = SoftPredictionUpload("b", np.array([[0.4, 0.6], [0.7, 0.3]], dtype=np.float32))
    mean = aggregate_mean([upload_a, upload_b])
    assert np.allclose(mean, [[0.3, 0.7], [0.6, 0.4]])


def test_aggregate_mean_idempotent_under_duplicate_client() -> None:
    upload_a = SoftPredictionUpload("a", np.array([[0.2, 0.8]], dtype=np.float32))
    upload_b = SoftPredictionUpload("b", np.array([[0.4, 0.6]], dtype=np.float32))
    once = aggregate_mean([upload_a, upload_b])
    duplicated = aggregate_mean([upload_a, upload_a, upload_b])
    assert np.allclose(once, duplicated)


def test_aggregate_mean_bit_identical_regardless_of_upload_order() -> None:
    """Ray/Flower reply-arrival order isn't reproducible run-to-run; aggregate_mean must sort by
    client_id internally so two runs with the same clients (in any arrival order) produce the
    exact same bytes -- not just approximately close (this was a real determinism bug, found by a
    live two-run integration test: T=0.1 sharpening amplifies tiny arrival-order-dependent
    floating-point summation differences into materially different training targets)."""
    rng = np.random.default_rng(0)
    uploads = [SoftPredictionUpload(str(i), rng.random((3, 11), dtype=np.float32)) for i in range(10)]
    forward = aggregate_mean(uploads)
    backward = aggregate_mean(list(reversed(uploads)))
    shuffled = aggregate_mean([uploads[i] for i in rng.permutation(len(uploads))])
    assert np.array_equal(forward, backward)
    assert np.array_equal(forward, shuffled)


def test_sharpen_rows_sum_to_one() -> None:
    probs = np.array([[0.9, 0.1], [0.5, 0.5]], dtype=np.float32)
    sharpened = sharpen(probs, temperature=0.1)
    assert np.allclose(sharpened.sum(axis=1), 1.0)


def test_sharpen_low_temperature_increases_confidence() -> None:
    probs = np.array([[0.9, 0.1]], dtype=np.float32)
    sharpened = sharpen(probs, temperature=0.1)
    assert sharpened[0, 0] > 0.9


def test_sharpen_temperature_one_is_near_identity() -> None:
    probs = np.array([[0.9, 0.1], [0.5, 0.5]], dtype=np.float32)
    sharpened = sharpen(probs, temperature=1.0)
    assert np.allclose(sharpened, probs, atol=1e-5)


def test_distill_step() -> None:
    model = build_classifier(Backbone.cnn)
    open_ds = _dataset(16, seed=1, labeled=False)
    rng = np.random.default_rng(0)
    targets = rng.random((16, NUM_CLASSES)).astype(np.float32)
    targets /= targets.sum(axis=1, keepdims=True)

    result = distill_step(model, open_ds, targets, DEVICE, epochs=1, lr=1e-3, batch_size=8, seed=0)
    assert result.epoch_losses
    assert math.isfinite(result.final_loss)


def test_server_evaluate() -> None:
    model = build_classifier(Backbone.cnn)
    test_ds = _dataset(20, seed=2, labeled=True)
    metrics = server_evaluate(model, test_ds, DEVICE, batch_size=8, seed=0)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert math.isfinite(metrics["loss"])
