"""CSV loading and per-file content validation for N-BaIoT source files.

One pyarrow read per file feeds both validation and sampling so the ~7.6GB raw dataset is only
read from disk once, not once to validate and again to sample.
"""

from __future__ import annotations

import numpy as np
import pyarrow.csv as pv

from ssfl.data.discovery import SourceFile
from ssfl.data.labels import NUM_FEATURES


class DataValidationError(ValueError):
    """Raised for any CSV content problem, with an actionable message identifying the file."""


def load_reference_feature_columns(input_path) -> list[str] | None:
    """Canonical feature name order from ``features.csv`` (Feature Name, Feature Description), if
    present. Used only as a stricter cross-check; discovery still falls back to the first file's
    header when this is absent."""
    features_csv = input_path / "features.csv"
    if not features_csv.exists():
        return None
    names: list[str] = []
    with open(features_csv, "r") as fh:
        header = fh.readline()
        if "Feature Name" not in header:
            return None
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            names.append(line.split(",", 1)[0])
    return names or None


def load_source_matrix(source: SourceFile) -> tuple[np.ndarray, list[str]]:
    """Read one source CSV into a ``(num_rows, 115)`` float32 matrix, preserving column order."""
    table = pv.read_csv(source.path)
    columns = table.column_names
    arrays = []
    for col in table.columns:
        arrays.append(col.to_numpy(zero_copy_only=False).astype(np.float32, copy=False))
    matrix = np.column_stack(arrays) if arrays else np.empty((table.num_rows, 0), dtype=np.float32)
    return matrix, columns


def validate_source_file(
    source: SourceFile,
    matrix: np.ndarray,
    columns: list[str],
    reference_columns: list[str],
    min_rows: int,
) -> None:
    """Validate one already-loaded source file: schema, finiteness, minimum row count."""
    where = f"device={source.device_id} class={source.class_key!r} ({source.path})"

    if len(columns) != NUM_FEATURES:
        raise DataValidationError(
            f"{where}: expected {NUM_FEATURES} feature columns, found {len(columns)}"
        )
    if len(set(columns)) != len(columns):
        seen: set[str] = set()
        dupes = [c for c in columns if c in seen or seen.add(c)]
        raise DataValidationError(f"{where}: duplicate column names {sorted(set(dupes))}")
    if columns != reference_columns:
        raise DataValidationError(
            f"{where}: feature column order/names differ from reference schema "
            f"(first mismatch at index {_first_mismatch(columns, reference_columns)})"
        )
    if matrix.shape[0] < min_rows:
        raise DataValidationError(
            f"{where}: only {matrix.shape[0]} rows, need at least {min_rows}"
        )
    if not np.isfinite(matrix).all():
        bad = int((~np.isfinite(matrix)).sum())
        raise DataValidationError(f"{where}: {bad} non-finite (NaN/Inf) values found")


def _first_mismatch(a: list[str], b: list[str]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))
