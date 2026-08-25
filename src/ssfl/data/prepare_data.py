"""Standalone N-BaIoT -> mini-N-BaIoT preparation CLI.

Architectural rule (M2): this module and everything it imports from ``ssfl.data`` must never
import Flower or training code. It only depends on ``ssfl.config`` for the plain pydantic
``DataPrepConfig``/``NormalizationMode`` types, which themselves do not import torch/Flower at
module scope (see ``ssfl.config.capture_environment_snapshot``, which imports torch lazily inside
its own function body).

Usage::

    uv run python -m ssfl.data.prepare_data --input data --output artifacts/data --seed 2023
    uv run python -m ssfl.data.prepare_data --input data --validate-only
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ssfl.config import DataPrepConfig, NormalizationMode
from ssfl.data.archive import ensure_extracted
from ssfl.data.discovery import DataDiscoveryError, SourceFile, discover_source_files, validate_discovery
from ssfl.data.io import DataValidationError, load_reference_feature_columns, load_source_matrix, validate_source_file
from ssfl.data.labels import LABEL_MAP
from ssfl.data.manifest import (
    atomic_write_json,
    checksum_tree,
    compute_allocation_stats,
    deterministic_savez,
    plot_allocation,
    sha256_json,
)
from ssfl.data.partition import build_scenario
from ssfl.data.sampling import AuditRow, SubsetSplit, sample_and_split
from ssfl.data.scaling import fit_scaler, reshape_eq19

NUM_LABELS = len(LABEL_MAP)


def _resolve_input(config: DataPrepConfig) -> tuple[Path, Path | None]:
    if config.input_path.is_dir():
        return config.input_path, None
    extract_dir = Path(tempfile.mkdtemp(prefix="ssfl-extract-"))
    resolved = ensure_extracted(config.input_path, extract_dir)
    return resolved, extract_dir


def _tool_version() -> dict[str, Any]:
    try:
        from importlib.metadata import version

        pkg_version = version("ssfl")
    except Exception:
        pkg_version = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
    except Exception:
        commit = None
    return {"ssfl_version": pkg_version, "git_commit": commit}


# ---------------------------------------------------------------------------
# --validate-only
# ---------------------------------------------------------------------------


def run_validate_only(config: DataPrepConfig) -> dict[str, Any]:
    input_dir, extract_dir = _resolve_input(config)
    try:
        files = discover_source_files(input_dir)
        validate_discovery(files)
        reference_columns = load_reference_feature_columns(input_dir)

        file_reports = []
        for source in files:
            matrix, columns = load_source_matrix(source)
            if reference_columns is None:
                reference_columns = columns
            validate_source_file(source, matrix, columns, reference_columns, config.samples_per_subset)
            file_reports.append(
                {
                    "device_id": source.device_id,
                    "class_key": source.class_key,
                    "num_rows": int(matrix.shape[0]),
                }
            )

        return {
            "status": "ok",
            "num_files": len(files),
            "num_devices": len({f.device_id for f in files}),
            "files": file_reports,
        }
    finally:
        if extract_dir is not None:
            shutil.rmtree(extract_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Full preparation
# ---------------------------------------------------------------------------


def _write_audit_parquet(
    path: Path,
    private_rows: list[AuditRow],
    open_rows_sorted: list[AuditRow],
    test_rows_sorted: list[AuditRow],
) -> None:
    records: list[dict[str, Any]] = []
    for r in sorted(private_rows, key=lambda r: (r.device_id, r.label, r.position)):
        records.append(
            {
                "device_id": r.device_id,
                "label": r.label,
                "split": r.split,
                "position": r.position,
                "source_row": r.source_row,
                "source_path": r.source_path,
                "global_index": None,
            }
        )
    for i, r in enumerate(open_rows_sorted):
        records.append(
            {
                "device_id": r.device_id,
                "label": r.label,
                "split": r.split,
                "position": r.position,
                "source_row": r.source_row,
                "source_path": r.source_path,
                "global_index": i,
            }
        )
    for i, r in enumerate(test_rows_sorted):
        records.append(
            {
                "device_id": r.device_id,
                "label": r.label,
                "split": r.split,
                "position": r.position,
                "source_row": r.source_row,
                "source_path": r.source_path,
                "global_index": i,
            }
        )
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path)


def _atomic_swap(staging: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    backup = output_path.parent / f"{output_path.name}.previous"
    if backup.exists():
        shutil.rmtree(backup)
    if output_path.exists():
        output_path.rename(backup)
    try:
        staging.rename(output_path)
    except Exception:
        if backup.exists():
            if output_path.exists():
                shutil.rmtree(output_path)
            backup.rename(output_path)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def run_full(config: DataPrepConfig) -> dict[str, Any]:
    input_dir, extract_dir = _resolve_input(config)
    try:
        files = discover_source_files(input_dir)
        validate_discovery(files)
        reference_columns = load_reference_feature_columns(input_dir)

        splits: dict[tuple[int, int], SubsetSplit] = {}
        audit_rows: list[AuditRow] = []
        devices: dict[int, list[int]] = {}

        for source in files:
            matrix, columns = load_source_matrix(source)
            if reference_columns is None:
                reference_columns = columns
            validate_source_file(source, matrix, columns, reference_columns, config.samples_per_subset)
            split = sample_and_split(source, matrix, config)
            splits[(source.device_id, source.label)] = split
            audit_rows.extend(split.audit_rows)
            devices.setdefault(source.device_id, []).append(source.label)

            rows = [r.source_row for r in split.audit_rows]
            if len(set(rows)) != len(rows):
                raise DataValidationError(
                    f"device={source.device_id} class={source.class_key!r}: "
                    "duplicate source row sampled within one subset"
                )

        n_private, n_open, n_test = config.split_counts
        expected_total = len(files) * config.samples_per_subset
        expected_private = len(files) * n_private
        expected_open = len(files) * n_open
        expected_test = len(files) * n_test

        actual_private = sum(s.private.shape[0] for s in splits.values())
        actual_open = sum(s.open.shape[0] for s in splits.values())
        actual_test = sum(s.test.shape[0] for s in splits.values())
        actual_total = actual_private + actual_open + actual_test
        if (actual_total, actual_private, actual_open, actual_test) != (
            expected_total,
            expected_private,
            expected_open,
            expected_test,
        ):
            raise DataValidationError(
                f"split totals mismatch: got total={actual_total} private={actual_private} "
                f"open={actual_open} test={actual_test}, expected total={expected_total} "
                f"private={expected_private} open={expected_open} test={expected_test}"
            )

        if config.normalization_mode == NormalizationMode.all_mini:
            fit_matrix = np.concatenate(
                [np.concatenate([s.private, s.open, s.test], axis=0) for s in splits.values()], axis=0
            )
        else:
            fit_matrix = np.concatenate([s.private for s in splits.values()], axis=0)
        scaler = fit_scaler(fit_matrix)

        staging = config.output_path.parent / f"{config.output_path.name}.building"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        for sub in ("private", "open", "test", "scenarios", "audit", "plots"):
            (staging / sub).mkdir()

        for (device_id, label), split in sorted(splits.items()):
            scaled = scaler.transform(split.private)
            deterministic_savez(
                staging / "private" / f"{device_id}_{label}.npz",
                features_flat=scaled.astype(np.float32),
                features_reshaped=reshape_eq19(scaled).astype(np.float32),
                label=np.int64(label),
                device_id=np.int64(device_id),
            )

        open_rows_sorted = sorted(
            (r for r in audit_rows if r.split == "open"), key=lambda r: (r.device_id, r.label, r.position)
        )
        open_matrix_raw = np.stack(
            [splits[(r.device_id, r.label)].open[r.position] for r in open_rows_sorted]
        )
        open_matrix = scaler.transform(open_matrix_raw)
        np.save(staging / "open" / "features.npy", open_matrix.astype(np.float32))
        np.save(staging / "open" / "features_reshaped.npy", reshape_eq19(open_matrix).astype(np.float32))

        test_rows_sorted = sorted(
            (r for r in audit_rows if r.split == "test"), key=lambda r: (r.device_id, r.label, r.position)
        )
        test_matrix_raw = np.stack(
            [splits[(r.device_id, r.label)].test[r.position] for r in test_rows_sorted]
        )
        test_matrix = scaler.transform(test_matrix_raw)
        test_labels = np.array([r.label for r in test_rows_sorted], dtype=np.int64)
        np.save(staging / "test" / "features.npy", test_matrix.astype(np.float32))
        np.save(staging / "test" / "features_reshaped.npy", reshape_eq19(test_matrix).astype(np.float32))
        np.save(staging / "test" / "labels.npy", test_labels)

        private_rows = [r for r in audit_rows if r.split == "private"]
        _write_audit_parquet(staging / "audit" / "source_rows.parquet", private_rows, open_rows_sorted, test_rows_sorted)

        deterministic_savez(
            staging / "scaler.npz",
            min=scaler.min_,
            max=scaler.max_,
            constant_mask=scaler.constant_mask,
            mode=np.array(config.normalization_mode.value),
        )

        atomic_write_json(
            staging / "feature_schema.json",
            {
                "num_features": len(reference_columns),
                "feature_names": reference_columns,
                "reshape_rows": 23,
                "reshape_cols": 5,
                "schema_hash": sha256_json(reference_columns),
            },
        )
        atomic_write_json(staging / "label_map.json", LABEL_MAP)

        scenario_client_counts: dict[str, int] = {}
        for scenario in (1, 2, 3):
            assignments = build_scenario(scenario, devices, n_private, config.seed, config.dirichlet_alpha)
            scenario_client_counts[str(scenario)] = len(assignments)
            atomic_write_json(
                staging / "scenarios" / f"{scenario}.json",
                {
                    "scenario": scenario,
                    "num_clients": len(assignments),
                    "seed": config.seed,
                    "clients": [
                        {
                            "client_id": a.client_id,
                            "device_id": a.device_id,
                            "num_examples": a.num_examples,
                            "class_local_indices": {str(k): v for k, v in a.class_local_indices.items()},
                        }
                        for a in assignments
                    ],
                },
            )
            stats = compute_allocation_stats(assignments, NUM_LABELS)
            atomic_write_json(staging / "scenarios" / f"{scenario}_allocation_stats.json", stats)
            plot_allocation(stats, scenario, staging / "plots" / f"scenario_{scenario}_allocation.png")

        # Computed before preparation_report.json (which carries a wall-clock timestamp) so that
        # report is excluded from the checksum ledger and manifest_hash -- otherwise re-running
        # with identical inputs/seed would produce a different hash on every run.
        checksums = checksum_tree(staging)
        manifest_body = {
            "seed": config.seed,
            "samples_per_subset": config.samples_per_subset,
            "split_counts": {"private": n_private, "open": n_open, "test": n_test},
            "total_records": actual_total,
            "private_records": actual_private,
            "open_records": actual_open,
            "test_records": actual_test,
            "num_devices": len(devices),
            "num_source_files": len(files),
            "device_class_map": {str(d): sorted(labels) for d, labels in devices.items()},
            "normalization_mode": config.normalization_mode.value,
            "dirichlet_alpha": config.dirichlet_alpha,
            "feature_schema_hash": sha256_json(reference_columns),
            "label_map": LABEL_MAP,
            "scenario_client_counts": scenario_client_counts,
            "checksums": checksums,
            "tool_version": _tool_version(),
        }
        manifest_hash = sha256_json(manifest_body)
        manifest_body["manifest_hash"] = manifest_hash
        manifest_body["created_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(staging / "dataset_manifest.json", manifest_body)

        atomic_write_json(
            staging / "preparation_report.json",
            {
                "prepared_at": datetime.now(timezone.utc).isoformat(),
                "seed": config.seed,
                "num_source_files": len(files),
                "num_devices": len(devices),
                "device_class_map": {str(d): sorted(labels) for d, labels in devices.items()},
                "split_counts": {"private": n_private, "open": n_open, "test": n_test},
                "total_records": actual_total,
                "private_records": actual_private,
                "open_records": actual_open,
                "test_records": actual_test,
                "normalization_mode": config.normalization_mode.value,
                "constant_feature_count": int(scaler.constant_mask.sum()),
                "scenario_client_counts": scenario_client_counts,
                "manifest_hash": manifest_hash,
            },
        )

        _atomic_swap(staging, config.output_path)
        return manifest_body
    finally:
        if extract_dir is not None:
            shutil.rmtree(extract_dir, ignore_errors=True)


def run(config: DataPrepConfig) -> dict[str, Any]:
    if config.validate_only:
        return run_validate_only(config)
    return run_full(config)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare mini-N-BaIoT from raw N-BaIoT CSVs.")
    parser.add_argument("--input", required=True, type=Path, dest="input_path")
    parser.add_argument("--output", type=Path, default=Path("artifacts/data"), dest="output_path")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--samples-per-subset", type=int, default=1000)
    parser.add_argument("--private-ratio", type=float, default=0.7)
    parser.add_argument("--open-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--normalization-mode",
        choices=[m.value for m in NormalizationMode],
        default=NormalizationMode.all_mini.value,
    )
    parser.add_argument("--dirichlet-alpha", type=float, default=0.1)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = DataPrepConfig(
        input_path=args.input_path,
        output_path=args.output_path,
        seed=args.seed,
        samples_per_subset=args.samples_per_subset,
        private_ratio=args.private_ratio,
        open_ratio=args.open_ratio,
        test_ratio=args.test_ratio,
        normalization_mode=NormalizationMode(args.normalization_mode),
        dirichlet_alpha=args.dirichlet_alpha,
        validate_only=args.validate_only,
    )
    try:
        result = run(config)
    except (DataDiscoveryError, DataValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
