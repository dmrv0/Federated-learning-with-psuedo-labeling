"""Strategy-level defenses: unauthorized senders and malformed payloads must be dropped rather
than crashing or corrupting aggregation. ``test_payload_limits.py`` covers the pure validators in
isolation; these tests prove the strategies actually call them on the real Flower ``Message`` path.
"""

import numpy as np
from flwr.app.metadata import Metadata
from flwr.common import Message, MetricRecord, RecordDict

from ssfl.config import VotingMode
from ssfl.records import array_record_from_numpy
from ssfl.strategies.dsfl import DSFLStrategy
from ssfl.strategies.fd import FDStrategy
from ssfl.strategies.ssfl import SSFLStrategy

NUM_CLASSES = 11
NUM_OPEN = 4


def _reply(arrays: dict, metrics: dict, src_node_id: int) -> Message:
    meta = Metadata(
        run_id=0,
        message_id=f"m{src_node_id}",
        src_node_id=src_node_id,
        dst_node_id=0,
        reply_to_message_id="",
        group_id="",
        created_at=0.0,
        ttl=100.0,
        message_type="train",
    )
    content = RecordDict(
        {"arrays": array_record_from_numpy(arrays), "metrics": MetricRecord(metrics)}
    )
    return Message(content=content, metadata=meta)


def _good_ssfl_arrays():
    return {"pseudo_labels": np.array([2, -1, 0, 10], dtype=np.int8)}


def _ssfl_metrics():
    return {"threshold": 0.5, "classifier_loss": 0.1, "discriminator_loss": 0.2}


def test_ssfl_drops_reply_from_unsampled_sender():
    strategy = SSFLStrategy(
        scenario=1,
        dataset_manifest_hash="h",
        num_open=NUM_OPEN,
        num_clients=2,
        voting_mode=VotingMode.enabled,
    )
    strategy._current_node_ids = [1, 2]
    intruder = _reply(_good_ssfl_arrays(), _ssfl_metrics(), src_node_id=999)
    arrays_out, metrics_out = strategy.aggregate_train(server_round=1, replies=[intruder])
    assert arrays_out is None and metrics_out is None


def test_ssfl_drops_malformed_payload_but_keeps_valid_ones():
    strategy = SSFLStrategy(
        scenario=1,
        dataset_manifest_hash="h",
        num_open=NUM_OPEN,
        num_clients=2,
        voting_mode=VotingMode.enabled,
    )
    strategy._current_node_ids = [1, 2]
    good = _reply(_good_ssfl_arrays(), _ssfl_metrics(), src_node_id=1)
    malformed = _reply(
        {
            "confidences": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
            "pseudo_labels": np.array([2, -1, 0, 10], dtype=np.int8),
        },
        _ssfl_metrics(),
        src_node_id=2,
    )
    arrays_out, metrics_out = strategy.aggregate_train(server_round=1, replies=[good, malformed])
    assert arrays_out is not None
    assert int(metrics_out["num_proposals"]) == 1


def _good_fd_arrays():
    class_probs = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float32)
    class_probs[0, 0] = 1.0
    class_present = np.zeros(NUM_CLASSES, dtype=np.int32)
    class_present[0] = 1
    return {"class_probs": class_probs, "class_present": class_present}


def test_fd_drops_reply_from_unsampled_sender():
    strategy = FDStrategy(num_clients=2)
    strategy._current_node_ids = [1, 2]
    intruder = _reply(_good_fd_arrays(), {"num-examples": 10.0}, src_node_id=999)
    arrays_out, metrics_out = strategy.aggregate_train(server_round=1, replies=[intruder])
    assert arrays_out is None and metrics_out is None


def test_fd_drops_malformed_payload():
    strategy = FDStrategy(num_clients=2)
    strategy._current_node_ids = [1, 2]
    bad = _good_fd_arrays()
    bad["class_present"][1] = 7  # not 0/1
    malformed = _reply(bad, {"num-examples": 10.0}, src_node_id=1)
    arrays_out, metrics_out = strategy.aggregate_train(server_round=1, replies=[malformed])
    assert arrays_out is None and metrics_out is None


def test_dsfl_drops_reply_from_unsampled_sender():
    strategy = DSFLStrategy(temperature=0.1, num_clients=2, num_open=NUM_OPEN)
    strategy._current_node_ids = [1, 2]
    probs = np.full((NUM_OPEN, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float32)
    intruder = _reply({"probs": probs}, {"num-examples": 10.0}, src_node_id=999)
    arrays_out, metrics_out = strategy.aggregate_train(server_round=1, replies=[intruder])
    assert arrays_out is None and metrics_out is None


def test_dsfl_drops_malformed_payload():
    strategy = DSFLStrategy(temperature=0.1, num_clients=2, num_open=NUM_OPEN)
    strategy._current_node_ids = [1, 2]
    bad_probs = np.zeros((NUM_OPEN, NUM_CLASSES), dtype=np.float32)
    bad_probs[:, 0] = 0.5  # rows don't sum to 1
    malformed = _reply({"probs": bad_probs}, {"num-examples": 10.0}, src_node_id=1)
    arrays_out, metrics_out = strategy.aggregate_train(server_round=1, replies=[malformed])
    assert arrays_out is None and metrics_out is None


def test_all_strategies_survive_zero_replies():
    """Missing-clients failure mode: every sampled client drops off the round entirely (crashed,
    network partition, timeout) so ``aggregate_train``/``aggregate_evaluate`` see an empty reply
    list. Must degrade to (None, None)/None, not raise -- the same path already proven for
    all-rejected replies above, exercised here for the genuinely-empty-list case directly."""
    ssfl = SSFLStrategy(
        scenario=1,
        dataset_manifest_hash="h",
        num_open=NUM_OPEN,
        num_clients=2,
        voting_mode=VotingMode.enabled,
    )
    ssfl._current_node_ids = [1, 2]
    assert ssfl.aggregate_train(server_round=1, replies=[]) == (None, None)
    assert ssfl.aggregate_evaluate(server_round=1, replies=[]) is None

    fd = FDStrategy(num_clients=2)
    fd._current_node_ids = [1, 2]
    assert fd.aggregate_train(server_round=1, replies=[]) == (None, None)

    dsfl = DSFLStrategy(temperature=0.1, num_clients=2, num_open=NUM_OPEN)
    dsfl._current_node_ids = [1, 2]
    assert dsfl.aggregate_train(server_round=1, replies=[]) == (None, None)


def test_ssfl_survives_all_clients_skipping_empty_distillation():
    strategy = SSFLStrategy(
        scenario=1,
        dataset_manifest_hash="h",
        num_open=NUM_OPEN,
        num_clients=2,
        voting_mode=VotingMode.enabled,
    )
    strategy._current_node_ids = [1, 2]
    replies = [
        _reply(
            {},
            {"loss": 0.0, "num-examples": 0, "distillation_skipped": 1},
            src_node_id=node_id,
        )
        for node_id in (1, 2)
    ]

    metrics = strategy.aggregate_evaluate(server_round=1, replies=replies)

    assert metrics["distillation_skipped"] == 1
    assert metrics["distillation_examples"] == 0
    assert metrics["responding_clients"] == 2
