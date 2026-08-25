"""Custom ``Strategy`` for FD: per-class averaged soft predictions with leave-self-out teacher
targets, matching ``protocols/fd.py`` exactly.

Unlike SSFL/DS-FL, the evaluate (distillation) exchange is per-client personalized -- each client's
teacher target excludes its own contribution -- so ``configure_evaluate`` builds one distinct
``Message`` per node instead of broadcasting a single shared payload.
"""

from __future__ import annotations

from collections.abc import Iterable

from flwr.common import ArrayRecord, ConfigRecord, Message, MessageType, MetricRecord, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import Strategy
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords, sample_nodes

from ssfl.models import NUM_CLASSES
from ssfl.protocols.fd import (
    ClassLogitUpload,
    FDAggregation,
    aggregate_class_logits,
    leave_self_out_targets,
)
from ssfl.protocols.message import ProtocolError
from ssfl.protocols.payload_limits import validate_fd_arrays
from ssfl.records import array_record_from_numpy, numpy_from_array_record


class FDStrategy(Strategy):
    def __init__(self, num_clients: int) -> None:
        self.num_clients = num_clients
        self._current_node_ids: list[int] = []
        self._last_aggregation: FDAggregation | None = None
        self._last_uploads: dict[str, ClassLogitUpload] = {}

    def summary(self) -> None:
        pass

    def configure_train(
        self, server_round: int, arrays, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        node_ids, _ = sample_nodes(grid, self.num_clients, self.num_clients)
        self._current_node_ids = node_ids
        config["server-round"] = server_round
        # ponytail: local training never needs model params or the prior round's leave-self-out
        # targets (clients only read config["server-round"]); empty ArrayRecord avoids re-shipping
        # the previous evaluate phase's global_sum/contributor_counts on every train message.
        record = RecordDict({"arrays": ArrayRecord(), "config": config})
        return [Message(record, message_type=MessageType.TRAIN, dst_node_id=n) for n in node_ids]

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        valid_senders = frozenset(str(n) for n in self._current_node_ids)
        replies = list(replies)
        uploads: list[ClassLogitUpload] = []
        contents = []
        for msg in replies:
            if msg.has_error():
                continue
            sender_id = str(msg.metadata.src_node_id)
            if sender_id not in valid_senders:
                continue
            arrays = numpy_from_array_record(msg.content["arrays"])
            try:
                validate_fd_arrays(arrays, num_classes=NUM_CLASSES)
            except ProtocolError:
                continue
            uploads.append(
                ClassLogitUpload(
                    client_id=sender_id,
                    class_probs=arrays["class_probs"],
                    class_present=arrays["class_present"],
                )
            )
            contents.append(msg.content)

        if not uploads:
            self._last_aggregation = None
            self._last_uploads = {}
            return None, None

        aggregation = aggregate_class_logits(uploads)
        self._last_aggregation = aggregation
        self._last_uploads = {u.client_id: u for u in uploads}

        arrays_out = array_record_from_numpy(
            {
                "global_sum": aggregation.global_sum,
                "contributor_counts": aggregation.contributor_counts,
            }
        )
        metrics_out = aggregate_metricrecords(contents, "num-examples")
        metrics_out["rejected_count"] = len(replies) - len(uploads)
        return arrays_out, metrics_out

    def configure_evaluate(
        self, server_round: int, arrays, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        config["server-round"] = server_round
        messages: list[Message] = []
        for node_id in self._current_node_ids:
            upload = self._last_uploads.get(str(node_id))
            if upload is None or self._last_aggregation is None:
                continue
            targets, valid = leave_self_out_targets(self._last_aggregation, upload)
            record = RecordDict(
                {
                    "arrays": array_record_from_numpy({"targets": targets, "valid": valid}),
                    "config": config,
                }
            )
            messages.append(Message(record, message_type=MessageType.EVALUATE, dst_node_id=node_id))
        return messages

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]):
        valid_senders = frozenset(str(n) for n in self._current_node_ids)
        valid_replies = [
            msg
            for msg in replies
            if not msg.has_error() and str(msg.metadata.src_node_id) in valid_senders
        ]
        if not valid_replies:
            return None
        # The paper explicitly describes FD's reported result as the highest-performing client,
        # not an invented global model or a private-data weighted average.
        best = max(valid_replies, key=lambda msg: float(msg.content["metrics"]["test_accuracy"]))
        metrics = best.content["metrics"]
        return MetricRecord(
            {
                "loss": float(metrics["test_loss"]),
                "accuracy": float(metrics["test_accuracy"]),
                "macro_precision": float(metrics["test_macro_precision"]),
                "macro_recall": float(metrics["test_macro_recall"]),
                "macro_f1": float(metrics["test_macro_f1"]),
                "selected_client": int(best.metadata.src_node_id),
                "num_replies": len(valid_replies),
            }
        )
