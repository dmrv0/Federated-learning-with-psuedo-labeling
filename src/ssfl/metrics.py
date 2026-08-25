"""Centralized classification metrics (accuracy, macro P/R/F1, confusion matrix) plus a
per-round ledger written to ``metrics.parquet``.

Pseudo-label accuracy against true open-set labels is intentionally NOT computed here -- that
would require reading labels the SSFL protocol must never expose to training (see
REPRODUCIBILITY.md). Any such audit belongs in the M8 reporting layer, which reads the sealed
``audit/source_rows.parquet`` separately from run artifacts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    confusion_matrix: np.ndarray
    micro_precision: float
    micro_recall: float
    micro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    per_class_precision: np.ndarray
    per_class_recall: np.ndarray
    per_class_f1: np.ndarray
    per_class_support: np.ndarray

    def to_dict(self, *, include_arrays: bool = False) -> dict:
        values = {
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "micro_precision": self.micro_precision,
            "micro_recall": self.micro_recall,
            "micro_f1": self.micro_f1,
            "weighted_precision": self.weighted_precision,
            "weighted_recall": self.weighted_recall,
            "weighted_f1": self.weighted_f1,
        }
        if include_arrays:
            values.update(
                {
                    "per_class_precision": self.per_class_precision.tolist(),
                    "per_class_recall": self.per_class_recall.tolist(),
                    "per_class_f1": self.per_class_f1.tolist(),
                    "per_class_support": self.per_class_support.tolist(),
                    "confusion_matrix": self.confusion_matrix.tolist(),
                }
            )
        return values


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> ClassificationMetrics:
    labels = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    micro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="micro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else float("nan")
    return ClassificationMetrics(
        accuracy=accuracy,
        macro_precision=float(macro[0]),
        macro_recall=float(macro[1]),
        macro_f1=float(macro[2]),
        confusion_matrix=cm,
        micro_precision=float(micro[0]),
        micro_recall=float(micro[1]),
        micro_f1=float(micro[2]),
        weighted_precision=float(weighted[0]),
        weighted_recall=float(weighted[1]),
        weighted_f1=float(weighted[2]),
        per_class_precision=precision,
        per_class_recall=recall,
        per_class_f1=f1,
        per_class_support=support,
    )


@dataclass
class MetricsLedger:
    """One row per centralized-eval call. Confusion matrices aren't scalar so they're kept out of
    the parquet table and flushed to a sidecar ``confusion_matrices.npz`` keyed by round."""

    rows: list[dict] = field(default_factory=list)
    _confusion_matrices: dict[int, np.ndarray] = field(default_factory=dict)
    per_class_rows: list[dict] = field(default_factory=list)
    run_dir: Path | None = None
    load_existing: bool = False
    completed_through: int | None = None

    def __post_init__(self) -> None:
        if self.run_dir is None or not self.load_existing:
            return
        metrics_path = self.run_dir / "metrics.parquet"
        if metrics_path.exists():
            import pandas as pd

            self.rows.extend(pd.read_parquet(metrics_path).to_dict("records"))
        per_class_path = self.run_dir / "per_class_metrics.parquet"
        if per_class_path.exists():
            import pandas as pd

            self.per_class_rows.extend(pd.read_parquet(per_class_path).to_dict("records"))
        confusion_path = self.run_dir / "confusion_matrices.npz"
        if confusion_path.exists():
            with np.load(confusion_path) as archive:
                for key in archive.files:
                    if key.startswith("round_"):
                        self._confusion_matrices[int(key.split("_")[1])] = archive[key]
        if self.completed_through is not None:
            self.rows = [row for row in self.rows if int(row["round"]) <= self.completed_through]
            self.per_class_rows = [
                row for row in self.per_class_rows if int(row["round"]) <= self.completed_through
            ]
            self._confusion_matrices = {
                round_number: matrix
                for round_number, matrix in self._confusion_matrices.items()
                if round_number <= self.completed_through
            }

    def record(
        self,
        *,
        algorithm: str,
        scenario: int,
        round: int,
        loss: float,
        train_loss: float | None = None,
        metrics: ClassificationMetrics | None = None,
        valid_rate: float | None = None,
    ) -> None:
        self._replace_round(algorithm, scenario, round)
        row = {
            "algorithm": algorithm,
            "scenario": scenario,
            "round": round,
            "loss": loss,
            "train_loss": train_loss,
            "valid_rate": valid_rate,
            "wall_clock": time.time(),
        }
        if metrics is not None:
            row.update(
                {
                    "accuracy": metrics.accuracy,
                    "macro_precision": metrics.macro_precision,
                    "macro_recall": metrics.macro_recall,
                    "macro_f1": metrics.macro_f1,
                    "micro_precision": metrics.micro_precision,
                    "micro_recall": metrics.micro_recall,
                    "micro_f1": metrics.micro_f1,
                    "weighted_precision": metrics.weighted_precision,
                    "weighted_recall": metrics.weighted_recall,
                    "weighted_f1": metrics.weighted_f1,
                }
            )
            self._confusion_matrices[round] = metrics.confusion_matrix
            for class_index in range(len(metrics.per_class_support)):
                self.per_class_rows.append(
                    {
                        "algorithm": algorithm,
                        "scenario": scenario,
                        "round": round,
                        "class": class_index,
                        "precision": float(metrics.per_class_precision[class_index]),
                        "recall": float(metrics.per_class_recall[class_index]),
                        "f1": float(metrics.per_class_f1[class_index]),
                        "support": int(metrics.per_class_support[class_index]),
                    }
                )
        self.rows.append(row)
        if self.run_dir is not None:
            self.write(self.run_dir)

    def record_precomputed(
        self,
        *,
        algorithm: str,
        scenario: int,
        round: int,
        loss: float,
        accuracy: float,
        macro_precision: float,
        macro_recall: float,
        macro_f1: float,
        selected_client: int | None = None,
    ) -> None:
        self._replace_round(algorithm, scenario, round)
        self.rows.append(
            {
                "algorithm": algorithm,
                "scenario": scenario,
                "round": round,
                "loss": loss,
                "accuracy": accuracy,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "selected_client": selected_client,
                "wall_clock": time.time(),
            }
        )
        if self.run_dir is not None:
            self.write(self.run_dir)

    def _replace_round(self, algorithm: str, scenario: int, round: int) -> None:
        """Make resume/retry writes idempotent for a given experiment round."""
        key = (algorithm, scenario, round)
        self.rows = [
            row
            for row in self.rows
            if (row.get("algorithm"), row.get("scenario"), row.get("round")) != key
        ]
        self.per_class_rows = [
            row
            for row in self.per_class_rows
            if (row.get("algorithm"), row.get("scenario"), row.get("round")) != key
        ]
        self._confusion_matrices.pop(round, None)

    def write(self, run_dir: Path) -> None:
        import pandas as pd

        metrics_path = run_dir / "metrics.parquet"
        metrics_tmp = metrics_path.with_suffix(".parquet.tmp")
        pd.DataFrame(self.rows).to_parquet(metrics_tmp, index=False)
        metrics_tmp.replace(metrics_path)
        per_class_path = run_dir / "per_class_metrics.parquet"
        per_class_tmp = per_class_path.with_suffix(".parquet.tmp")
        pd.DataFrame(self.per_class_rows).to_parquet(per_class_tmp, index=False)
        per_class_tmp.replace(per_class_path)
        if self._confusion_matrices:
            confusion_path = run_dir / "confusion_matrices.npz"
            confusion_tmp = confusion_path.with_suffix(".tmp.npz")
            np.savez(
                confusion_tmp, **{f"round_{r}": cm for r, cm in self._confusion_matrices.items()}
            )
            confusion_tmp.replace(confusion_path)


if __name__ == "__main__":
    y_true = np.array([0, 1, 1, 2, 0])
    y_pred = np.array([0, 1, 0, 2, 0])
    m = compute_classification_metrics(y_true, y_pred, num_classes=3)
    assert 0.0 <= m.accuracy <= 1.0 and abs(m.accuracy - 4 / 5) < 1e-9
    assert m.confusion_matrix.shape == (3, 3)
    assert m.confusion_matrix.sum() == len(y_true)

    ledger = MetricsLedger()
    ledger.record(algorithm="ssfl", scenario=1, round=1, loss=0.5, metrics=m)
    assert ledger.rows[0]["accuracy"] == m.accuracy
    assert 1 in ledger._confusion_matrices
    print("metrics.py self-check OK")
