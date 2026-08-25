"""Custom ``Strategy`` for DS-FL: soft open-set predictions -> mean -> temperature sharpen ->
broadcast -> distillation, matching ``protocols/dsfl.py`` exactly. Only soft prediction vectors and
sharpened targets cross the wire -- never model parameters.
"""

from __future__ import annotations

from collections.abc import Iterable

from flwr.common import ArrayRecord, ConfigRecord, Message, MessageType, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import Strategy
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords, sample_nodes

from ssfl.models import NUM_CLASSES
from ssfl.protocols.dsfl import SoftPredictionUpload, aggregate_mean, sharpen
from ssfl.protocols.message import ProtocolError
from ssfl.protocols.payload_limits import validate_dsfl_arrays
from ssfl.records import array_record_from_numpy, numpy_from_array_record


class DSFLStrategy(Strategy):
    def __init__(self, temperature: float, num_clients: int, num_open: int) -> None:
        self.temperature = temperature
        self.num_clients = num_clients
        self.num_open = num_open
        self._current_node_ids: list[int] = []

    def summary(self) -> None:
        pass

    def configure_train(
        self, server_round: int, arrays, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        node_ids, _ = sample_nodes(grid, self.num_clients, self.num_clients)
        self._current_node_ids = node_ids
        config["server-round"] = server_round
        # ponytail: local training never needs model params or the prior round's sharpened
        # targets (clients only read config["server-round"]); empty ArrayRecord avoids re-shipping
        # the previous evaluate phase's sharpened_targets on every train message.
        record = RecordDict({"arrays": ArrayRecord(), "config": config})
        return [Message(record, message_type=MessageType.TRAIN, dst_node_id=n) for n in node_ids]

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        valid_senders = frozenset(str(n) for n in self._current_node_ids)
        replies = list(replies)
        uploads: list[SoftPredictionUpload] = []
        contents = []
        for msg in replies:
            if msg.has_error():
                continue
            sender_id = str(msg.metadata.src_node_id)
            if sender_id not in valid_senders:
                continue
            arrays = numpy_from_array_record(msg.content["arrays"])
            try:
                validate_dsfl_arrays(arrays, num_open=self.num_open, num_classes=NUM_CLASSES)
            except ProtocolError:
                continue
            uploads.append(SoftPredictionUpload(client_id=sender_id, probs=arrays["probs"]))
            contents.append(msg.content)

        if not uploads:
            return None, None

        mean_probs = aggregate_mean(uploads)
        sharpened = sharpen(mean_probs, temperature=self.temperature)
        arrays_out = array_record_from_numpy({"sharpened_targets": sharpened})
        metrics_out = aggregate_metricrecords(contents, "num-examples")
        metrics_out["rejected_count"] = len(replies) - len(uploads)
        return arrays_out, metrics_out

    def configure_evaluate(
        self, server_round: int, arrays, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        config["server-round"] = server_round
        record = RecordDict({"arrays": arrays, "config": config})
        return [Message(record, message_type=MessageType.EVALUATE, dst_node_id=n) for n in self._current_node_ids]

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]):
        valid_senders = frozenset(str(n) for n in self._current_node_ids)
        contents = [
            msg.content for msg in replies if not msg.has_error() and str(msg.metadata.src_node_id) in valid_senders
        ]
        if not contents:
            return None
        return aggregate_metricrecords(contents, "num-examples")
