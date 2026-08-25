from dataclasses import asdict

import numpy as np
import pandas as pd

from ssfl.comms import CommsLedger, CommsRow
from ssfl.metrics import MetricsLedger


def _comms_row(round_number: int) -> CommsRow:
    return CommsRow(
        attempt_id="old-attempt",
        timestamp_ns=round_number,
        algorithm="ssfl",
        scenario=1,
        round=round_number,
        phase="train",
        direction="client_to_server",
        sender="1",
        recipient="server",
        tensor_names="arrays.pseudo_labels",
        dtypes="int8",
        shapes="4",
        logical_bytes=4,
        serialized_bytes=20,
        paper_bytes=4,
    )


def test_resume_comms_discards_uncommitted_rounds(tmp_path) -> None:
    path = tmp_path / "communication.parquet"
    pd.DataFrame([asdict(_comms_row(1)), asdict(_comms_row(2))]).to_parquet(path, index=False)

    ledger = CommsLedger(
        path=path,
        attempt_id="new-attempt",
        load_existing=True,
        completed_through=1,
    )

    assert [row.round for row in ledger.rows] == [1]


def test_resume_metrics_discards_uncommitted_rounds(tmp_path) -> None:
    pd.DataFrame(
        [
            {"algorithm": "ssfl", "scenario": 1, "round": 1, "loss": 1.0},
            {"algorithm": "ssfl", "scenario": 1, "round": 2, "loss": 2.0},
        ]
    ).to_parquet(tmp_path / "metrics.parquet", index=False)
    pd.DataFrame(
        [
            {"algorithm": "ssfl", "scenario": 1, "round": 1, "class": 0},
            {"algorithm": "ssfl", "scenario": 1, "round": 2, "class": 0},
        ]
    ).to_parquet(tmp_path / "per_class_metrics.parquet", index=False)
    np.savez(
        tmp_path / "confusion_matrices.npz",
        round_1=np.eye(2, dtype=np.int64),
        round_2=np.eye(2, dtype=np.int64),
    )

    ledger = MetricsLedger(
        run_dir=tmp_path,
        load_existing=True,
        completed_through=1,
    )

    assert [row["round"] for row in ledger.rows] == [1]
    assert [row["round"] for row in ledger.per_class_rows] == [1]
    assert list(ledger._confusion_matrices) == [1]
