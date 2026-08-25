"""SSFL protocol: two-phase per-round exchange (proposal + distillation).

Pure functions/dataclasses over models, tensors, and datasets -- no Flower dependency, per the M4
gate. Flower ``ClientApp``/``ServerApp`` wiring (M5) calls straight into this module. Callers keep
holding the same classifier/discriminator ``nn.Module`` instances across rounds (in-process here,
via ``Context.state`` once wired to Flower) -- that is what gives "persistent private client
models" for free at this layer, without this module needing to know how state is stored.

Privacy boundary (REPRODUCIBILITY.md / DATA_CARD.md): :class:`ProposalResult` -- the only thing a
client returns from the proposal phase -- carries pseudo-labels and scalar metrics only. It never
contains model parameters, gradients, private features, or optimizer state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import TensorDataset

from ssfl.config import DiscriminatorMode, LabelRepresentation, ThresholdPolicy
from ssfl.models import SSFLModel
from ssfl.protocols.message import Envelope
from ssfl.training import TrainResult, evaluate, make_loader, predict_probs, train_supervised
from ssfl.telemetry import EventCallback, gpu_snapshot

ABSTAIN = -1


@dataclass(frozen=True)
class ProposalResult:
    client_id: str
    # Exactly one of pseudo_labels/soft_probs is set, matching ssfl_label_representation:
    # hard -> pseudo_labels (int8, ABSTAIN(-1) or class index); soft -> soft_probs (float32,
    # (num_open, num_classes), an all-zero row standing in for "unfamiliar" -- softmax output
    # never legitimately sums to 0, so a zero row is unambiguous).
    pseudo_labels: np.ndarray | None
    confidences: np.ndarray | None  # client-local audit only; never serialized by canonical SSFL
    threshold: float
    classifier_loss: float
    discriminator_loss: float | None  # None when discriminator_mode != enabled (never trained)
    soft_probs: np.ndarray | None = None


def compute_threshold(confidences: np.ndarray, policy: ThresholdPolicy) -> float:
    if policy == ThresholdPolicy.median:
        return float(np.median(confidences))
    value = policy.fixed_value
    assert value is not None, f"non-median policy {policy} must define a fixed_value"
    return value


def client_proposal_step(
    client_id: str,
    classifier: SSFLModel,
    discriminator: SSFLModel,
    private_dataset,
    open_dataset,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    threshold_policy: ThresholdPolicy,
    seed: int,
    discriminator_mode: DiscriminatorMode = DiscriminatorMode.enabled,
    label_representation: LabelRepresentation = LabelRepresentation.hard,
    soft_round_decimals: int | None = None,
    event_callback: EventCallback | None = None,
) -> ProposalResult:
    """Phase A: train classifier on private data, score the open set, then decide which open
    examples are "familiar" one of three ways (``discriminator_mode``) and encode the result one
    of two ways (``label_representation``) -- the two independent axes the paper's five ablations
    are combinations of (full / no-discriminator / no-voting / no-discriminator-or-voting /
    simple-filtering; see ``REPRODUCIBILITY.md``)."""
    private_loader = make_loader(private_dataset, batch_size, shuffle=True, seed=seed)
    classifier_result = train_supervised(
        classifier,
        private_loader,
        device,
        epochs,
        lr,
        event_callback=event_callback,
        stage="client_classifier_supervised",
    )

    open_loader = make_loader(open_dataset, batch_size, shuffle=False, seed=seed)
    open_probs = predict_probs(
        classifier,
        open_loader,
        device,
        event_callback=event_callback,
        stage="client_open_classifier_prediction",
    )
    confidences = open_probs.max(dim=1).values.numpy()
    threshold = compute_threshold(confidences, threshold_policy)
    class_predictions = open_probs.argmax(dim=1).numpy()

    discriminator_loss: float | None = None
    if discriminator_mode == DiscriminatorMode.disabled:
        familiar_mask = np.ones(len(confidences), dtype=bool)
    elif discriminator_mode == DiscriminatorMode.simple_filter:
        familiar_mask = confidences >= threshold
    else:
        # Discriminator targets: private examples are "familiar" (class 0), low-confidence open
        # examples are "unfamiliar" (class 1) -- the paper's discriminator dataset construction.
        unfamiliar_mask = confidences < threshold
        unfamiliar_x = open_dataset.x[torch.from_numpy(unfamiliar_mask)]
        disc_x = torch.cat([private_dataset.x, unfamiliar_x], dim=0)
        disc_y = torch.cat(
            [
                torch.zeros(private_dataset.x.shape[0], dtype=torch.long),
                torch.ones(unfamiliar_x.shape[0], dtype=torch.long),
            ]
        )
        disc_loader = make_loader(
            TensorDataset(disc_x, disc_y), batch_size, shuffle=True, seed=seed
        )
        discriminator_result = train_supervised(
            discriminator,
            disc_loader,
            device,
            epochs,
            lr,
            event_callback=event_callback,
            stage="client_discriminator_supervised",
        )
        familiar_probs = predict_probs(
            discriminator,
            open_loader,
            device,
            event_callback=event_callback,
            stage="client_open_discriminator_prediction",
        )
        familiar_mask = familiar_probs.argmax(dim=1).numpy() == 0
        discriminator_loss = discriminator_result.final_loss

    if label_representation == LabelRepresentation.hard:
        pseudo_labels = np.where(familiar_mask, class_predictions, ABSTAIN).astype(np.int8)
        soft_probs = None
    else:
        soft = np.where(familiar_mask[:, None], open_probs.numpy(), 0.0).astype(np.float32)
        if soft_round_decimals is not None:
            soft = np.round(soft, decimals=soft_round_decimals).astype(np.float32)
        pseudo_labels = None
        soft_probs = soft

    if event_callback:
        counts = np.bincount(class_predictions, minlength=open_probs.shape[1])
        event_callback(
            "proposal_summary",
            {
                "threshold": threshold,
                "confidence_min": float(confidences.min()),
                "confidence_q05": float(np.quantile(confidences, 0.05)),
                "confidence_q25": float(np.quantile(confidences, 0.25)),
                "confidence_median": float(np.median(confidences)),
                "confidence_q75": float(np.quantile(confidences, 0.75)),
                "confidence_q95": float(np.quantile(confidences, 0.95)),
                "confidence_max": float(confidences.max()),
                "confidence_mean": float(confidences.mean()),
                "confidence_std": float(confidences.std()),
                "familiar_count": int(familiar_mask.sum()),
                "unfamiliar_count": int((~familiar_mask).sum()),
                "familiar_rate": float(familiar_mask.mean()),
                "prediction_class_counts": counts.tolist(),
                "classifier_final_loss": classifier_result.final_loss,
                "discriminator_final_loss": discriminator_loss,
            },
        )

    return ProposalResult(
        client_id=client_id,
        pseudo_labels=pseudo_labels,
        soft_probs=soft_probs,
        confidences=confidences.astype(np.float32),
        threshold=threshold,
        classifier_loss=classifier_result.final_loss,
        discriminator_loss=discriminator_loss,
    )


@dataclass(frozen=True)
class AggregationResult:
    global_labels: np.ndarray  # int64, len num_open, ABSTAIN(-1) where invalid
    valid_mask: np.ndarray  # bool, len num_open
    votes_per_class: np.ndarray  # (num_open, num_classes) int64, for audit
    participating_counts: np.ndarray  # (num_open,) int64, non-abstaining voters per index
    tie_count: int
    all_abstain_count: int
    rejected: tuple[tuple[str, str], ...]  # (sender_id, reason) for entries dropped pre-vote


def aggregate_votes(
    proposals: list[tuple[Envelope, ProposalResult]], num_open: int, num_classes: int
) -> AggregationResult:
    """Per-index majority vote; ties -> lowest class index; all-abstain -> ABSTAIN + invalid.

    Idempotent by construction: a duplicated envelope for a sender already counted is dropped
    (into ``rejected``) rather than counted twice, so re-aggregating a batch that accidentally
    contains a retried message produces the same result as aggregating it once.
    """
    seen_senders: set[str] = set()
    votes = np.zeros((num_open, num_classes), dtype=np.int64)
    rejected: list[tuple[str, str]] = []
    for envelope, result in proposals:
        if envelope.sender_id in seen_senders:
            rejected.append((envelope.sender_id, "duplicate sender in aggregation batch"))
            continue
        seen_senders.add(envelope.sender_id)
        labels = result.pseudo_labels
        idx = np.nonzero(labels != ABSTAIN)[0]
        votes[idx, labels[idx]] += 1

    global_labels = np.full(num_open, ABSTAIN, dtype=np.int64)
    valid_mask = np.zeros(num_open, dtype=bool)
    tie_count = 0
    all_abstain_count = 0
    for i in range(num_open):
        row = votes[i]
        total = int(row.sum())
        if total == 0:
            all_abstain_count += 1
            continue
        max_votes = row.max()
        winners = np.nonzero(row == max_votes)[0]
        if len(winners) > 1:
            tie_count += 1
        global_labels[i] = int(winners.min())
        valid_mask[i] = True

    return AggregationResult(
        global_labels=global_labels,
        valid_mask=valid_mask,
        votes_per_class=votes,
        participating_counts=votes.sum(axis=1),
        tie_count=tie_count,
        all_abstain_count=all_abstain_count,
        rejected=tuple(rejected),
    )


def aggregate_soft(
    proposals: list[tuple[Envelope, ProposalResult]], num_open: int, num_classes: int
) -> AggregationResult:
    """No-voting variant (``ssfl_voting_mode=disabled``): mean the masked soft probability
    vectors (an all-zero row = that client found the example unfamiliar) across clients per open
    index, then argmax -> hard global label. Same idempotent-under-duplicate-sender behavior as
    ``aggregate_votes``. ``votes_per_class`` doesn't apply to a soft mean, so it's left zeroed
    rather than repurposed to hold something misleading.

    Summed in ``sender_id``-sorted order rather than reply-arrival order: floating-point addition
    is not associative, and Ray/Flower reply arrival order is not guaranteed reproducible across
    runs of the same simulation, so an arrival-order sum would make this ablation's results
    non-deterministic across otherwise-identical seeded runs."""
    seen_senders: set[str] = set()
    sum_probs = np.zeros((num_open, num_classes), dtype=np.float64)
    counts = np.zeros(num_open, dtype=np.int64)
    rejected: list[tuple[str, str]] = []
    for envelope, result in sorted(proposals, key=lambda p: p[0].sender_id):
        if envelope.sender_id in seen_senders:
            rejected.append((envelope.sender_id, "duplicate sender in aggregation batch"))
            continue
        seen_senders.add(envelope.sender_id)
        probs = result.soft_probs
        mask = probs.sum(axis=1) > 0
        sum_probs[mask] += probs[mask]
        counts[mask] += 1

    valid_mask = counts > 0
    global_labels = np.full(num_open, ABSTAIN, dtype=np.int64)
    idx = np.nonzero(valid_mask)[0]
    if len(idx):
        mean_probs = (sum_probs[idx] / counts[idx, None]).astype(np.float32)
        global_labels[idx] = mean_probs.argmax(axis=1)

    return AggregationResult(
        global_labels=global_labels,
        valid_mask=valid_mask,
        votes_per_class=np.zeros((num_open, num_classes), dtype=np.int64),
        participating_counts=counts,
        tie_count=0,
        all_abstain_count=int((~valid_mask).sum()),
        rejected=tuple(rejected),
    )


def _valid_open_dataset(open_dataset, aggregation: AggregationResult) -> TensorDataset:
    valid_idx = np.nonzero(aggregation.valid_mask)[0]
    x = open_dataset.x[torch.from_numpy(valid_idx)]
    y = torch.from_numpy(aggregation.global_labels[valid_idx]).long()
    return TensorDataset(x, y)


def client_distillation_step(
    classifier: SSFLModel,
    open_dataset,
    aggregation: AggregationResult,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    event_callback: EventCallback | None = None,
):
    """Phase B (client): hard-label CE on the valid subset of the global open labels. Returns
    metrics only -- no model parameters cross this function's boundary back to a message."""
    dataset = _valid_open_dataset(open_dataset, aggregation)
    if len(dataset) == 0:
        if event_callback:
            event_callback(
                "training_skipped",
                {
                    "stage": "client_global_label_distillation",
                    "reason": "no_globally_valid_pseudo_labels",
                    "epochs_requested": epochs,
                    "total_examples": 0,
                    "total_batches": 0,
                    **gpu_snapshot(),
                },
            )
        return TrainResult()
    loader = make_loader(dataset, batch_size, shuffle=True, seed=seed)
    return train_supervised(
        classifier,
        loader,
        device,
        epochs,
        lr,
        event_callback=event_callback,
        stage="client_global_label_distillation",
    )


def server_distillation_step(
    server_classifier: SSFLModel,
    open_dataset,
    aggregation: AggregationResult,
    test_dataset,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    event_callback: EventCallback | None = None,
):
    """Phase B (server): train the server's own persistent classifier on the same valid open
    labels, then evaluate it on the full test set."""
    dataset = _valid_open_dataset(open_dataset, aggregation)
    if len(dataset) == 0:
        train_result = TrainResult()
        if event_callback:
            event_callback(
                "training_skipped",
                {
                    "stage": "server_global_label_distillation",
                    "reason": "no_globally_valid_pseudo_labels",
                    "epochs_requested": epochs,
                    "total_examples": 0,
                    "total_batches": 0,
                    **gpu_snapshot(),
                },
            )
    else:
        loader = make_loader(dataset, batch_size, shuffle=True, seed=seed)
        train_result = train_supervised(
            server_classifier,
            loader,
            device,
            epochs,
            lr,
            event_callback=event_callback,
            stage="server_global_label_distillation",
        )

    test_loader = make_loader(test_dataset, batch_size, shuffle=False, seed=seed)
    eval_metrics = evaluate(
        server_classifier,
        test_loader,
        device,
        event_callback=event_callback,
        stage="server_test_evaluation",
    )
    return train_result, eval_metrics
