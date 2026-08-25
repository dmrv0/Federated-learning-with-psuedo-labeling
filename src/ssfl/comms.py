"""Per-message communication accounting.

``CommsTrackingStrategy`` wraps any concrete ``Strategy`` (custom or stock ``FedAvg``) and
intercepts every ``configure_train``/``aggregate_train``/``configure_evaluate``/
``aggregate_evaluate`` call -- the only places a ``Message`` crosses the client/server wire in
Flower's Message API. Wrapping there means every algorithm gets identical, complete accounting
without instrumenting each protocol's aggregation logic individually, and centralized
``evaluate_fn`` calls (which never leave the ServerApp process) are correctly excluded.

Three byte counts are recorded per message, matching REPRODUCIBILITY.md's guidance to never
force-match a single number to the paper:
  - ``logical_bytes``: raw ndarray payload only (what the algorithm actually needed to send).
  - ``serialized_bytes``: Flower's actual protobuf wire encoding (framing/type overhead included).
  - ``paper_bytes``: the paper's dtype/accounting convention for Table IV comparisons.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
import time
from typing import Any

from flwr.common import Message, RecordDict
from flwr.common.serde import recorddict_to_proto
from flwr.serverapp.strategy import Strategy
from flwr.serverapp.strategy.result import Result

from ssfl.logging_utils import log_event
from ssfl.telemetry import JsonlEventWriter, gpu_snapshot


@dataclass(frozen=True)
class CommsRow:
    attempt_id: str
    timestamp_ns: int
    algorithm: str
    scenario: int
    round: int
    phase: str
    direction: str
    sender: str
    recipient: str
    tensor_names: str
    dtypes: str
    shapes: str
    logical_bytes: int
    serialized_bytes: int
    paper_bytes: int


def _paper_array_bytes(name: str, array: Any) -> int:
    """Paper Table IV uses one representative client's logical uplink payload.

    The paper explicitly describes soft labels as double precision and hard labels as one label
    value, so retain that convention independently from the real transport dtype.
    """
    if name.endswith("pseudo_labels") or name.endswith("global_labels"):
        return int(array.size)  # signed int8, including the -1 abstain sentinel
    if any(name.endswith(suffix) for suffix in ("soft_probs", "probs", "class_probs")):
        return int(array.size * 8)  # paper: double-valued probability vectors
    if name.endswith("class_present"):
        return 0  # implementation-only empty-class guard, absent from the paper's accounting
    return int(array.nbytes)


def record_dict_bytes(record: RecordDict) -> tuple[int, int, int, list[str], list[str], list[str]]:
    names, dtypes, shapes = [], [], []
    logical = 0
    paper = 0
    for record_name, array_record in record.array_records.items():
        for array_name, array in array_record.items():
            arr = array.numpy()
            logical += arr.nbytes
            full_name = f"{record_name}.{array_name}"
            paper += _paper_array_bytes(full_name, arr)
            names.append(full_name)
            dtypes.append(str(arr.dtype))
            shapes.append("x".join(str(d) for d in arr.shape))
    serialized = recorddict_to_proto(record).ByteSize()
    return logical, serialized, paper, names, dtypes, shapes


@dataclass
class CommsLedger:
    rows: list[CommsRow] = field(default_factory=list)
    path: Path | None = None
    attempt_id: str = "unknown"
    load_existing: bool = False
    completed_through: int | None = None

    def __post_init__(self) -> None:
        if self.load_existing and self.path is not None and self.path.exists():
            import pandas as pd

            for row in pd.read_parquet(self.path).to_dict("records"):
                row.setdefault("attempt_id", "legacy")
                row.setdefault("timestamp_ns", 0)
                row.setdefault("paper_bytes", row.get("logical_bytes", 0))
                self.rows.append(CommsRow(**row))
            if self.completed_through is not None:
                self.rows = [row for row in self.rows if row.round <= self.completed_through]

    def record_messages(
        self,
        messages: Iterable[Message],
        algorithm: str,
        scenario: int,
        round: int,
        phase: str,
        direction: str,
    ) -> None:
        for msg in messages:
            if msg.has_error():
                continue
            logical, serialized, paper, names, dtypes, shapes = record_dict_bytes(msg.content)
            if direction == "server_to_client":
                sender, recipient = "server", str(msg.metadata.dst_node_id)
            else:
                sender, recipient = str(msg.metadata.src_node_id), "server"
            self.rows.append(
                CommsRow(
                    attempt_id=self.attempt_id,
                    timestamp_ns=time.time_ns(),
                    algorithm=algorithm,
                    scenario=scenario,
                    round=round,
                    phase=phase,
                    direction=direction,
                    sender=sender,
                    recipient=recipient,
                    tensor_names=",".join(names),
                    dtypes=",".join(dtypes),
                    shapes=",".join(shapes),
                    logical_bytes=logical,
                    serialized_bytes=serialized,
                    paper_bytes=paper,
                )
            )
        if self.path is not None:
            self.write_parquet(self.path)

    def write_parquet(self, path: Path) -> None:
        import pandas as pd

        tmp = path.with_suffix(".parquet.tmp")
        pd.DataFrame([asdict(r) for r in self.rows]).to_parquet(tmp, index=False)
        tmp.replace(path)


class CommsTrackingStrategy(Strategy):
    """Delegates aggregation behavior to ``inner`` unchanged; only observes the ``Message``
    lists flowing through the four exchange points to populate ``self.ledger`` and, if ``logger``
    is given, to emit one structured ``events.jsonl`` line per round/phase (M9 "Logging"/"Metrics":
    round, phase, reply count, and every numeric scalar the inner strategy's aggregation returned --
    e.g. SSFL/FD/DS-FL's ``rejected_count``, SSFL's ``valid_rate``/``tie_count``. Never touches
    ``arrays_out``, so model weights/pseudo-labels can never end up in the log by construction)."""

    def __init__(
        self,
        inner: Strategy,
        algorithm: str,
        scenario: int,
        logger: Any = None,
        telemetry: JsonlEventWriter | None = None,
        ledger_path: Path | None = None,
        attempt_id: str = "unknown",
        load_existing: bool = False,
        completed_through: int | None = None,
        start_round: int = 1,
        round_end_callback: Callable[[int, Any, Result], None] | None = None,
    ) -> None:
        self.inner = inner
        self.algorithm = algorithm
        self.scenario = scenario
        self.ledger = CommsLedger(
            path=ledger_path,
            attempt_id=attempt_id,
            load_existing=load_existing,
            completed_through=completed_through,
        )
        self.logger = logger
        self.telemetry = telemetry
        self.start_round = start_round
        self.round_end_callback = round_end_callback

    def summary(self) -> None:
        self.inner.summary()

    def _log_round(self, server_round: int, phase: str, num_replies: int, metrics_out: Any) -> None:
        fields: dict[str, Any] = {"round": server_round, "phase": phase, "num_replies": num_replies}
        if metrics_out is not None:
            for key, value in metrics_out.items():
                if isinstance(value, (int, float, bool)):
                    fields[key] = value
        if self.logger is not None:
            log_event(self.logger, "aggregate", **fields)
        if self.telemetry is not None:
            self.telemetry.emit("aggregate", **fields, **gpu_snapshot())

    def _log_replies(self, server_round: int, phase: str, replies: list[Message]) -> None:
        if self.telemetry is None:
            return
        for reply in replies:
            fields: dict[str, Any] = {
                "round": server_round,
                "phase": phase,
                "sender": str(reply.metadata.src_node_id),
                "has_error": reply.has_error(),
            }
            if not reply.has_error():
                for record_name, metrics in reply.content.metric_records.items():
                    for key, value in metrics.items():
                        if isinstance(value, (str, int, float, bool)):
                            fields[f"{record_name}.{key}"] = value
                logical, serialized, paper, names, dtypes, shapes = record_dict_bytes(reply.content)
                fields.update(
                    {
                        "logical_bytes": logical,
                        "serialized_bytes": serialized,
                        "paper_bytes": paper,
                        "tensor_names": names,
                        "dtypes": dtypes,
                        "shapes": shapes,
                    }
                )
            self.telemetry.emit("client_reply", **fields)

    def configure_train(self, server_round, arrays, config, grid) -> list[Message]:
        messages = list(self.inner.configure_train(server_round, arrays, config, grid))
        self.ledger.record_messages(
            messages, self.algorithm, self.scenario, server_round, "train", "server_to_client"
        )
        return messages

    def aggregate_train(self, server_round, replies: Iterable[Message]):
        replies = list(replies)
        self._log_replies(server_round, "train", replies)
        self.ledger.record_messages(
            replies, self.algorithm, self.scenario, server_round, "train", "client_to_server"
        )
        arrays_out, metrics_out = self.inner.aggregate_train(server_round, replies)
        self._log_round(server_round, "train", len(replies), metrics_out)
        return arrays_out, metrics_out

    def configure_evaluate(self, server_round, arrays, config, grid) -> list[Message]:
        messages = list(self.inner.configure_evaluate(server_round, arrays, config, grid))
        self.ledger.record_messages(
            messages, self.algorithm, self.scenario, server_round, "evaluate", "server_to_client"
        )
        return messages

    def aggregate_evaluate(self, server_round, replies: Iterable[Message]):
        replies = list(replies)
        self._log_replies(server_round, "evaluate", replies)
        self.ledger.record_messages(
            replies, self.algorithm, self.scenario, server_round, "evaluate", "client_to_server"
        )
        metrics_out = self.inner.aggregate_evaluate(server_round, replies)
        self._log_round(server_round, "evaluate", len(replies), metrics_out)
        return metrics_out

    def start(
        self,
        grid,
        initial_arrays,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config=None,
        evaluate_config=None,
        evaluate_fn=None,
    ) -> Result:
        """Flower's round loop with durable phase telemetry and resume from a completed round."""
        from flwr.common import ConfigRecord

        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()
        arrays = initial_arrays
        if self.start_round == 1 and evaluate_fn:
            initial_metrics = evaluate_fn(0, arrays)
            if initial_metrics is not None:
                result.evaluate_metrics_serverapp[0] = initial_metrics

        for current_round in range(self.start_round, num_rounds + 1):
            round_started = time.perf_counter()
            if self.telemetry:
                self.telemetry.emit("round_start", round=current_round, **gpu_snapshot())

            phase_started = time.perf_counter()
            train_messages = self.configure_train(current_round, arrays, train_config, grid)
            train_replies = grid.send_and_receive(messages=train_messages, timeout=timeout)
            agg_arrays, train_metrics = self.aggregate_train(current_round, train_replies)
            if agg_arrays is not None:
                arrays = agg_arrays
                result.arrays = agg_arrays
            if train_metrics is not None:
                result.train_metrics_clientapp[current_round] = train_metrics
            if self.telemetry:
                self.telemetry.emit(
                    "phase_end",
                    round=current_round,
                    phase="proposal_or_local_train",
                    duration_seconds=time.perf_counter() - phase_started,
                    **gpu_snapshot(),
                )

            phase_started = time.perf_counter()
            evaluate_messages = self.configure_evaluate(
                current_round, arrays, evaluate_config, grid
            )
            evaluate_replies = grid.send_and_receive(messages=evaluate_messages, timeout=timeout)
            evaluate_metrics = self.aggregate_evaluate(current_round, evaluate_replies)
            if evaluate_metrics is not None:
                result.evaluate_metrics_clientapp[current_round] = evaluate_metrics
            if self.telemetry:
                self.telemetry.emit(
                    "phase_end",
                    round=current_round,
                    phase="distillation_or_client_evaluate",
                    duration_seconds=time.perf_counter() - phase_started,
                    **gpu_snapshot(),
                )

            if evaluate_fn:
                server_metrics = evaluate_fn(current_round, arrays)
                if server_metrics is not None:
                    result.evaluate_metrics_serverapp[current_round] = server_metrics

            if self.round_end_callback:
                self.round_end_callback(current_round, arrays, result)
            if self.telemetry:
                self.telemetry.emit(
                    "round_end",
                    round=current_round,
                    duration_seconds=time.perf_counter() - round_started,
                    **gpu_snapshot(),
                )
        return result


if __name__ == "__main__":
    import numpy as np
    from flwr.common import Array, ArrayRecord, MessageType

    rec = RecordDict(
        {
            "arrays": ArrayRecord(
                array_dict={"x": Array.from_numpy_ndarray(np.zeros((4, 5), dtype=np.float32))}
            )
        }
    )
    logical, serialized, paper, names, dtypes, shapes = record_dict_bytes(rec)
    assert logical == 4 * 5 * 4
    assert serialized > 0
    assert paper == logical
    assert names == ["arrays.x"]

    ledger = CommsLedger()
    msg = Message(rec, message_type=MessageType.TRAIN, dst_node_id=7)
    ledger.record_messages([msg], "ssfl", 1, 1, "train", "server_to_client")
    assert len(ledger.rows) == 1
    assert ledger.rows[0].recipient == "7"
    print("comms.py self-check OK")
