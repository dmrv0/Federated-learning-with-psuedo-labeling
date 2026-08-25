import dataclasses
import math

import numpy as np
import pytest
import torch

from ssfl.config import Algorithm, Backbone, DiscriminatorMode, LabelRepresentation, ThresholdPolicy
from ssfl.data.datasets import TensorFeatureDataset
from ssfl.models import build_classifier, build_discriminator
from ssfl.protocols.message import Envelope
from ssfl.protocols.ssfl import (
    ABSTAIN,
    AggregationResult,
    ProposalResult,
    aggregate_soft,
    aggregate_votes,
    client_distillation_step,
    client_proposal_step,
    compute_threshold,
    server_distillation_step,
)

DEVICE = torch.device("cpu")


def _dataset(n: int, seed: int, labeled: bool) -> TensorFeatureDataset:
    rng = np.random.default_rng(seed)
    flat = rng.random((n, 115), dtype=np.float64).astype(np.float32)
    reshaped = flat.reshape(n, 5, 23).transpose(0, 2, 1)
    labels = rng.integers(0, 11, size=n).astype(np.int64) if labeled else None
    return TensorFeatureDataset(flat, reshaped, Backbone.cnn, labels)


def _envelope(sender_id: str) -> Envelope:
    return Envelope(
        algorithm=Algorithm.ssfl,
        scenario=1,
        round=1,
        phase="proposal",
        sender_id=sender_id,
        dataset_manifest_hash="abc",
    )


def test_compute_threshold_median() -> None:
    confidences = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    assert compute_threshold(confidences, ThresholdPolicy.median) == 0.5


def test_compute_threshold_fixed() -> None:
    confidences = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    assert compute_threshold(confidences, ThresholdPolicy.fixed_0_8) == 0.8


def test_proposal_result_carries_no_private_state() -> None:
    """Privacy boundary: ProposalResult must only ever hold pseudo-labels/scalars -- never model
    parameters, gradients, or private feature tensors."""
    field_names = {f.name for f in dataclasses.fields(ProposalResult)}
    assert field_names == {
        "client_id",
        "pseudo_labels",
        "confidences",
        "threshold",
        "classifier_loss",
        "discriminator_loss",
        "soft_probs",
    }
    instance = ProposalResult("c0", np.zeros(1), np.zeros(1, np.float32), 0.5, 0.0, 0.0)
    for value in dataclasses.asdict(instance).values():
        assert not isinstance(value, torch.nn.Module)
        assert not isinstance(value, torch.Tensor)  # payload is numpy/scalar only, never a tensor


def test_client_proposal_step_end_to_end() -> None:
    classifier = build_classifier(Backbone.cnn)
    discriminator = build_discriminator(Backbone.cnn)
    private = _dataset(32, seed=1, labeled=True)
    open_ds = _dataset(16, seed=2, labeled=False)

    result = client_proposal_step(
        client_id="c0",
        classifier=classifier,
        discriminator=discriminator,
        private_dataset=private,
        open_dataset=open_ds,
        device=DEVICE,
        epochs=1,
        lr=1e-3,
        batch_size=8,
        threshold_policy=ThresholdPolicy.median,
        seed=0,
    )

    assert result.pseudo_labels.shape == (16,)
    assert set(np.unique(result.pseudo_labels)).issubset(set(range(-1, 11)))
    assert result.confidences.shape == (16,)
    assert math.isfinite(result.classifier_loss)
    assert math.isfinite(result.discriminator_loss)
    # every field is a plain scalar/array -- never a tensor requiring grad or a module
    assert isinstance(result.pseudo_labels, np.ndarray)
    assert isinstance(result.confidences, np.ndarray)


def test_client_proposal_step_discriminator_disabled_all_familiar() -> None:
    """no-discriminator ablation: every open example treated as familiar (no ABSTAIN)."""
    classifier = build_classifier(Backbone.cnn)
    discriminator = build_discriminator(Backbone.cnn)
    result = client_proposal_step(
        client_id="c0",
        classifier=classifier,
        discriminator=discriminator,
        private_dataset=_dataset(32, seed=1, labeled=True),
        open_dataset=_dataset(16, seed=2, labeled=False),
        device=DEVICE,
        epochs=1,
        lr=1e-3,
        batch_size=8,
        threshold_policy=ThresholdPolicy.median,
        seed=0,
        discriminator_mode=DiscriminatorMode.disabled,
    )
    assert result.discriminator_loss is None
    assert not np.any(result.pseudo_labels == ABSTAIN)


def test_client_proposal_step_simple_filter_no_discriminator_model() -> None:
    """simple-filtering ablation: confidence>=threshold decides familiarity, no discriminator training."""
    classifier = build_classifier(Backbone.cnn)
    discriminator = build_discriminator(Backbone.cnn)
    result = client_proposal_step(
        client_id="c0",
        classifier=classifier,
        discriminator=discriminator,
        private_dataset=_dataset(32, seed=1, labeled=True),
        open_dataset=_dataset(16, seed=2, labeled=False),
        device=DEVICE,
        epochs=1,
        lr=1e-3,
        batch_size=8,
        threshold_policy=ThresholdPolicy.median,
        seed=0,
        discriminator_mode=DiscriminatorMode.simple_filter,
    )
    assert result.discriminator_loss is None
    # median threshold -> exactly half familiar (ties broken by >=)
    assert (result.pseudo_labels != ABSTAIN).sum() >= 1


def test_client_proposal_step_soft_label_representation() -> None:
    """no-voting ablation: soft masked probability rows, rounded, unfamiliar rows all-zero."""
    classifier = build_classifier(Backbone.cnn)
    discriminator = build_discriminator(Backbone.cnn)
    result = client_proposal_step(
        client_id="c0",
        classifier=classifier,
        discriminator=discriminator,
        private_dataset=_dataset(32, seed=1, labeled=True),
        open_dataset=_dataset(16, seed=2, labeled=False),
        device=DEVICE,
        epochs=1,
        lr=1e-3,
        batch_size=8,
        threshold_policy=ThresholdPolicy.median,
        seed=0,
        label_representation=LabelRepresentation.soft,
        soft_round_decimals=4,
    )
    assert result.pseudo_labels is None
    assert result.soft_probs.shape == (16, 11)
    row_sums = result.soft_probs.sum(axis=1)
    for value in row_sums:
        assert value == pytest.approx(0.0, abs=1e-6) or value == pytest.approx(1.0, abs=1e-3)


def test_aggregate_soft_means_and_argmaxes() -> None:
    num_classes = 3
    probs_a = np.array([[0.7, 0.2, 0.1], [0.0, 0.0, 0.0]], dtype=np.float32)  # 2nd row unfamiliar
    probs_b = np.array(
        [[0.5, 0.3, 0.2], [0.0, 0.0, 0.0]], dtype=np.float32
    )  # both unfamiliar on row 2
    client_a = ProposalResult(
        "a", None, np.zeros(2, np.float32), 0.5, 0.0, None, soft_probs=probs_a
    )
    client_b = ProposalResult(
        "b", None, np.zeros(2, np.float32), 0.5, 0.0, None, soft_probs=probs_b
    )
    proposals = [(_envelope("a"), client_a), (_envelope("b"), client_b)]

    result = aggregate_soft(proposals, num_open=2, num_classes=num_classes)

    assert result.global_labels[0] == 0  # mean([.7,.2,.1],[.5,.3,.2]) argmax -> class 0
    assert result.valid_mask[0]
    assert result.global_labels[1] == ABSTAIN  # all clients found it unfamiliar
    assert not result.valid_mask[1]
    assert result.all_abstain_count == 1


def test_aggregate_soft_bit_identical_regardless_of_proposal_order() -> None:
    """Same determinism requirement as DS-FL's aggregate_mean / FD's aggregate_class_logits: Ray/
    Flower reply order isn't reproducible run-to-run, so summing soft_probs in reply order made
    the no-voting ablation's global_labels non-deterministic across identically-seeded runs. Fixed
    by sorting on sender_id before summing."""
    rng = np.random.default_rng(2)
    num_open, num_classes = 5, 4
    proposals = [
        (
            _envelope(str(i)),
            ProposalResult(
                str(i),
                None,
                np.zeros(num_open, np.float32),
                0.5,
                0.0,
                None,
                soft_probs=rng.random((num_open, num_classes), dtype=np.float32),
            ),
        )
        for i in range(10)
    ]
    forward = aggregate_soft(proposals, num_open=num_open, num_classes=num_classes)
    backward = aggregate_soft(list(reversed(proposals)), num_open=num_open, num_classes=num_classes)
    shuffled = aggregate_soft(
        [proposals[i] for i in rng.permutation(len(proposals))],
        num_open=num_open,
        num_classes=num_classes,
    )
    assert np.array_equal(forward.global_labels, backward.global_labels)
    assert np.array_equal(forward.global_labels, shuffled.global_labels)


def test_aggregate_votes_majority_tie_and_all_abstain() -> None:
    client_a = ProposalResult(
        "a", np.array([0, 0, ABSTAIN]), np.zeros(3, np.float32), 0.5, 0.0, 0.0
    )
    client_b = ProposalResult(
        "b", np.array([0, 1, ABSTAIN]), np.zeros(3, np.float32), 0.5, 0.0, 0.0
    )
    client_c = ProposalResult(
        "c", np.array([1, ABSTAIN, ABSTAIN]), np.zeros(3, np.float32), 0.5, 0.0, 0.0
    )
    proposals = [(_envelope("a"), client_a), (_envelope("b"), client_b), (_envelope("c"), client_c)]

    result = aggregate_votes(proposals, num_open=3, num_classes=3)

    assert result.global_labels[0] == 0  # 2 votes for 0 vs 1 for 1
    assert result.valid_mask[0]
    assert result.global_labels[1] == 0  # tie between 0 and 1 -> lowest index wins
    assert result.tie_count == 1
    assert result.valid_mask[1]
    assert result.global_labels[2] == ABSTAIN  # all three abstained
    assert not result.valid_mask[2]
    assert result.all_abstain_count == 1


def test_aggregate_votes_idempotent_under_duplicate_sender() -> None:
    client_a = ProposalResult("a", np.array([0]), np.zeros(1, np.float32), 0.5, 0.0, 0.0)
    once = aggregate_votes([(_envelope("a"), client_a)], num_open=1, num_classes=2)
    duplicated = aggregate_votes(
        [(_envelope("a"), client_a), (_envelope("a"), client_a)], num_open=1, num_classes=2
    )

    assert np.array_equal(once.global_labels, duplicated.global_labels)
    assert np.array_equal(once.votes_per_class, duplicated.votes_per_class)
    assert len(duplicated.rejected) == 1


def test_client_and_server_distillation_step() -> None:
    open_ds = _dataset(16, seed=3, labeled=False)
    global_labels = np.array([i % 11 for i in range(16)], dtype=np.int64)
    valid_mask = np.ones(16, dtype=bool)
    aggregation = AggregationResult(
        global_labels=global_labels,
        valid_mask=valid_mask,
        votes_per_class=np.zeros((16, 11), dtype=np.int64),
        participating_counts=np.ones(16, dtype=np.int64),
        tie_count=0,
        all_abstain_count=0,
        rejected=(),
    )

    classifier = build_classifier(Backbone.cnn)
    client_result = client_distillation_step(
        classifier, open_ds, aggregation, DEVICE, epochs=1, lr=1e-3, batch_size=8, seed=0
    )
    assert math.isfinite(client_result.final_loss)

    server_classifier = build_classifier(Backbone.cnn)
    test_ds = _dataset(20, seed=4, labeled=True)
    train_result, eval_metrics = server_distillation_step(
        server_classifier,
        open_ds,
        aggregation,
        test_ds,
        DEVICE,
        epochs=1,
        lr=1e-3,
        batch_size=8,
        seed=0,
    )
    assert math.isfinite(train_result.final_loss)
    assert 0.0 <= eval_metrics["accuracy"] <= 1.0
    assert math.isfinite(eval_metrics["loss"])


def test_client_and_server_skip_empty_distillation_but_server_still_evaluates() -> None:
    open_ds = _dataset(16, seed=3, labeled=False)
    aggregation = AggregationResult(
        global_labels=np.full(16, ABSTAIN, dtype=np.int64),
        valid_mask=np.zeros(16, dtype=bool),
        votes_per_class=np.zeros((16, 11), dtype=np.int64),
        participating_counts=np.zeros(16, dtype=np.int64),
        tie_count=0,
        all_abstain_count=16,
        rejected=(),
    )
    events: list[tuple[str, dict]] = []

    classifier = build_classifier(Backbone.cnn)
    before = {name: value.clone() for name, value in classifier.state_dict().items()}
    client_result = client_distillation_step(
        classifier,
        open_ds,
        aggregation,
        DEVICE,
        epochs=1,
        lr=1e-3,
        batch_size=8,
        seed=0,
        event_callback=lambda event, fields: events.append((event, fields)),
    )
    assert client_result.total_examples == 0
    assert math.isnan(client_result.final_loss)
    assert all(torch.equal(before[name], value) for name, value in classifier.state_dict().items())
    assert events[-1][0] == "training_skipped"
    assert events[-1][1]["reason"] == "no_globally_valid_pseudo_labels"

    server_classifier = build_classifier(Backbone.cnn)
    test_ds = _dataset(20, seed=4, labeled=True)
    train_result, eval_metrics = server_distillation_step(
        server_classifier,
        open_ds,
        aggregation,
        test_ds,
        DEVICE,
        epochs=1,
        lr=1e-3,
        batch_size=8,
        seed=0,
    )
    assert train_result.total_examples == 0
    assert 0.0 <= eval_metrics["accuracy"] <= 1.0
    assert math.isfinite(eval_metrics["loss"])
