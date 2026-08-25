import math

import numpy as np
import torch

from ssfl.config import Backbone
from ssfl.data.datasets import TensorFeatureDataset
from ssfl.models import NUM_CLASSES, build_classifier
from ssfl.protocols.fd import (
    ClassLogitUpload,
    aggregate_class_logits,
    client_distillation_step,
    leave_self_out_targets,
)

DEVICE = torch.device("cpu")


def _dataset(n: int, seed: int, label: int) -> TensorFeatureDataset:
    rng = np.random.default_rng(seed)
    flat = rng.random((n, 115), dtype=np.float64).astype(np.float32)
    reshaped = flat.reshape(n, 5, 23).transpose(0, 2, 1)
    labels = np.full(n, label, dtype=np.int64)
    return TensorFeatureDataset(flat, reshaped, Backbone.cnn, labels)


def _upload(client_id: str, present_classes: dict[int, np.ndarray]) -> ClassLogitUpload:
    class_probs = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float32)
    class_present = np.zeros(NUM_CLASSES, dtype=np.int32)
    for c, vec in present_classes.items():
        class_probs[c] = vec
        class_present[c] = 1
    return ClassLogitUpload(client_id, class_probs, class_present)


def test_aggregate_class_logits_sums_present_classes_only() -> None:
    vec0_a = np.eye(NUM_CLASSES, dtype=np.float32)[0]
    vec0_b = np.eye(NUM_CLASSES, dtype=np.float32)[1]
    vec1_c = np.eye(NUM_CLASSES, dtype=np.float32)[1]

    upload_a = _upload("a", {0: vec0_a})
    upload_b = _upload("b", {0: vec0_b})
    upload_c = _upload("c", {1: vec1_c})

    aggregation = aggregate_class_logits([upload_a, upload_b, upload_c])

    assert aggregation.contributor_counts[0] == 2
    assert aggregation.contributor_counts[1] == 1
    assert aggregation.contributor_counts[2] == 0
    assert np.allclose(aggregation.global_sum[0], vec0_a + vec0_b)
    assert np.allclose(aggregation.global_sum[1], vec1_c)


def test_aggregate_class_logits_idempotent_under_duplicate_client() -> None:
    upload_a = _upload("a", {0: np.eye(NUM_CLASSES, dtype=np.float32)[0]})
    once = aggregate_class_logits([upload_a])
    duplicated = aggregate_class_logits([upload_a, upload_a])
    assert np.array_equal(once.global_sum, duplicated.global_sum)
    assert np.array_equal(once.contributor_counts, duplicated.contributor_counts)


def test_aggregate_class_logits_bit_identical_regardless_of_upload_order() -> None:
    """Same determinism requirement as aggregate_mean (DS-FL): Ray/Flower reply order isn't
    reproducible run-to-run, so summing class_probs in upload order made global_sum -- and
    everything downstream (leave-self-out targets, distillation) -- non-deterministic. Fixed by
    sorting on client_id before summing."""
    rng = np.random.default_rng(1)
    uploads = [
        _upload(str(i), {c: rng.random(NUM_CLASSES, dtype=np.float32) for c in range(NUM_CLASSES)})
        for i in range(10)
    ]
    forward = aggregate_class_logits(uploads)
    backward = aggregate_class_logits(list(reversed(uploads)))
    shuffled = aggregate_class_logits([uploads[i] for i in rng.permutation(len(uploads))])
    assert np.array_equal(forward.global_sum, backward.global_sum)
    assert np.array_equal(forward.global_sum, shuffled.global_sum)
    assert np.array_equal(forward.contributor_counts, shuffled.contributor_counts)


def test_leave_self_out_targets_multi_contributor() -> None:
    vec0_a = np.eye(NUM_CLASSES, dtype=np.float32)[0]
    vec0_b = np.eye(NUM_CLASSES, dtype=np.float32)[1]
    upload_a = _upload("a", {0: vec0_a})
    upload_b = _upload("b", {0: vec0_b})
    aggregation = aggregate_class_logits([upload_a, upload_b])

    targets, valid = leave_self_out_targets(aggregation, upload_a)
    assert valid[0]
    assert np.allclose(targets[0], vec0_b)  # excludes upload_a's own contribution


def test_leave_self_out_targets_sole_contributor_is_invalid() -> None:
    vec1_c = np.eye(NUM_CLASSES, dtype=np.float32)[1]
    upload_c = _upload("c", {1: vec1_c})
    aggregation = aggregate_class_logits([upload_c])

    targets, valid = leave_self_out_targets(aggregation, upload_c)
    assert not valid[1]


def test_leave_self_out_targets_benefits_from_others_missing_class() -> None:
    vec0_a = np.eye(NUM_CLASSES, dtype=np.float32)[0]
    vec1_c = np.eye(NUM_CLASSES, dtype=np.float32)[1]
    upload_a = _upload("a", {0: vec0_a})
    upload_c = _upload("c", {1: vec1_c})
    aggregation = aggregate_class_logits([upload_a, upload_c])

    # upload_a never contributed class 1, but should still get c's contribution as its target
    targets, valid = leave_self_out_targets(aggregation, upload_a)
    assert valid[1]
    assert np.allclose(targets[1], vec1_c)


def test_leave_self_out_targets_missing_class_federation_wide_is_invalid() -> None:
    upload_a = _upload("a", {0: np.eye(NUM_CLASSES, dtype=np.float32)[0]})
    aggregation = aggregate_class_logits([upload_a])
    _, valid = leave_self_out_targets(aggregation, upload_a)
    assert not valid[5]  # no client federation-wide ever contributed class 5


def test_client_distillation_step_trains_on_valid_classes() -> None:
    rng = np.random.default_rng(0)
    targets = rng.random((NUM_CLASSES, NUM_CLASSES)).astype(np.float32)
    targets /= targets.sum(axis=1, keepdims=True)
    valid = np.zeros(NUM_CLASSES, dtype=bool)
    valid[0] = True

    classifier = build_classifier(Backbone.cnn)
    private = _dataset(16, seed=1, label=0)
    result = client_distillation_step(
        classifier, private, targets, valid, DEVICE, epochs=1, lr=1e-3, batch_size=8, seed=0
    )
    assert result.epoch_losses
    assert math.isfinite(result.final_loss)


def test_client_distillation_step_skips_when_no_valid_classes_present() -> None:
    targets = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float32)
    valid = np.zeros(NUM_CLASSES, dtype=bool)  # nothing valid anywhere

    classifier = build_classifier(Backbone.cnn)
    private = _dataset(16, seed=1, label=0)
    result = client_distillation_step(
        classifier, private, targets, valid, DEVICE, epochs=1, lr=1e-3, batch_size=8, seed=0
    )
    assert result.epoch_losses == []
    assert math.isnan(result.final_loss)
