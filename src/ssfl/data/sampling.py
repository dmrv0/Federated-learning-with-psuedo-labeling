"""Deterministic 1,000-row sampling and 700/100/200 split for one (device, class) source file."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ssfl.config import DataPrepConfig
from ssfl.data.discovery import SourceFile


@dataclass(frozen=True)
class AuditRow:
    device_id: int
    label: int
    split: str  # "private" | "open" | "test"
    position: int  # index within that split, 0-based
    source_row: int  # 0-indexed data row within the source CSV (header excluded)
    source_path: str


@dataclass(frozen=True)
class SubsetSplit:
    device_id: int
    label: int
    private: np.ndarray
    open: np.ndarray
    test: np.ndarray
    audit_rows: list[AuditRow]


def subset_seed(global_seed: int, device_id: int, label: int) -> int:
    """Deterministic per-(device,class) seed via ``SeedSequence`` entropy mixing rather than plain
    addition, so e.g. (device=2,label=3) and (device=3,label=2) never collide on the same seed."""
    seq = np.random.SeedSequence([global_seed, device_id, label])
    return int(seq.generate_state(1)[0])


def sample_and_split(source: SourceFile, matrix: np.ndarray, config: DataPrepConfig) -> SubsetSplit:
    """Draw ``samples_per_subset`` rows without replacement and partition them into
    private/open/test. Sampling and splitting share one seeded draw: the selection order from
    ``rng.choice`` is sliced directly into the three splits, which keeps the splits disjoint by
    construction and needs only one RNG stream per subset."""
    n_private, n_open, n_test = config.split_counts
    total = config.samples_per_subset
    if matrix.shape[0] < total:
        raise ValueError(
            f"device={source.device_id} class={source.class_key!r}: only {matrix.shape[0]} rows, "
            f"need at least {total} to sample"
        )

    seed = subset_seed(config.seed, source.device_id, source.label)
    rng = np.random.default_rng(seed)
    selected = rng.choice(matrix.shape[0], size=total, replace=False)

    private_idx = selected[:n_private]
    open_idx = selected[n_private : n_private + n_open]
    test_idx = selected[n_private + n_open :]

    audit_rows: list[AuditRow] = []
    for split_name, idx_array in (("private", private_idx), ("open", open_idx), ("test", test_idx)):
        for position, row_idx in enumerate(idx_array):
            audit_rows.append(
                AuditRow(
                    device_id=source.device_id,
                    label=source.label,
                    split=split_name,
                    position=position,
                    source_row=int(row_idx),
                    source_path=str(source.path),
                )
            )

    return SubsetSplit(
        device_id=source.device_id,
        label=source.label,
        private=matrix[private_idx],
        open=matrix[open_idx],
        test=matrix[test_idx],
        audit_rows=audit_rows,
    )
