"""Run-directory conventions and reproducibility-bundle scaffolding.

Every experiment run gets one directory under ``<output_path>/<run_id>/`` containing exactly the
layout the M10 reporting/reproducibility-bundle contract expects:

    resolved_config.yaml
    environment.json
    dataset_manifest.json
    code_version.json
    metrics.parquet          (created by the metrics writer, M6)
    communication.parquet    (created by the comms writer, M6)
    attempts/<attempt-id>/events.jsonl
    attempts/<attempt-id>/telemetry/{server.jsonl,clients/*.jsonl}
    checkpoints/
    plots/
    summary.json

``RunContext.create`` writes the identity files (config/environment/manifest/code-version) before
any training starts, satisfying "every resolved configuration must be written to the run directory
before training" and "record dataset hashes, configuration, dependency versions, random seeds,
device information ... before training begins".
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
import torch

from ssfl.config import ExperimentConfig, capture_environment_snapshot, compute_run_id


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str))


def prune_superseded_checkpoints(
    checkpoint_dir: Path, current_round: int, pinned_rounds: tuple[int, ...]
) -> list[Path]:
    """Keep milestone checkpoints plus the latest resumable checkpoint.

    Per-round training telemetry remains untouched; only older non-milestone model snapshots are
    removed after their replacement is safely written.
    """
    keep = {int(round_number) for round_number in pinned_rounds}
    keep.add(int(current_round))
    pruned: list[Path] = []
    for path in sorted(checkpoint_dir.glob("round_*.pt")):
        try:
            round_number = int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if round_number not in keep:
            path.unlink()
            pruned.append(path)
    return pruned


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    config: ExperimentConfig
    attempt_id: str
    attempt_dir: Path

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def plots_dir(self) -> Path:
        return self.run_dir / "plots"

    @property
    def events_path(self) -> Path:
        return self.attempt_dir / "events.jsonl"

    @property
    def telemetry_dir(self) -> Path:
        path = self.attempt_dir / "telemetry"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _create_attempt(run_dir: Path) -> tuple[str, Path]:
        attempt_id = f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
        attempt_dir = run_dir / "attempts" / attempt_id
        attempt_dir.mkdir(parents=True, exist_ok=False)
        _atomic_write_json(
            run_dir / "current_attempt.json",
            {"attempt_id": attempt_id, "attempt_dir": str(attempt_dir)},
        )
        return attempt_id, attempt_dir

    @classmethod
    def create(
        cls,
        config: ExperimentConfig,
        dataset_manifest_path: Path | None = None,
        code_version: dict[str, Any] | None = None,
    ) -> "RunContext":
        dataset_manifest_hash = None
        dataset_manifest: dict[str, Any] | None = None
        if dataset_manifest_path is not None and dataset_manifest_path.exists():
            dataset_manifest = json.loads(dataset_manifest_path.read_text())
            dataset_manifest_hash = dataset_manifest.get("manifest_hash")

        run_id = compute_run_id(config, dataset_manifest_hash=dataset_manifest_hash)
        run_dir = config.output_path / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        (run_dir / "plots").mkdir(exist_ok=True)
        attempt_id, attempt_dir = cls._create_attempt(run_dir)

        _atomic_write_text(
            run_dir / "resolved_config.yaml",
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True),
        )
        _atomic_write_json(run_dir / "environment.json", capture_environment_snapshot())
        if dataset_manifest is not None:
            _atomic_write_json(run_dir / "dataset_manifest.json", dataset_manifest)
        _atomic_write_json(run_dir / "code_version.json", code_version or {})

        return cls(
            run_id=run_id,
            run_dir=run_dir,
            config=config,
            attempt_id=attempt_id,
            attempt_dir=attempt_dir,
        )

    @classmethod
    def resume(cls, run_dir: Path) -> "RunContext":
        """Reload a previously created run directory for resume, revalidating that the resolved
        config on disk still round-trips through the current ``ExperimentConfig`` schema (a schema
        drift between the code that created the run and the code resuming it must fail loudly
        rather than resume with silently reinterpreted fields)."""
        config_path = run_dir / "resolved_config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"{run_dir} has no resolved_config.yaml; not a valid run dir")
        payload = yaml.safe_load(config_path.read_text())
        config = ExperimentConfig.model_validate(payload)
        attempt_id, attempt_dir = cls._create_attempt(run_dir)
        return cls(
            run_id=run_dir.name,
            run_dir=run_dir,
            config=config,
            attempt_id=attempt_id,
            attempt_dir=attempt_dir,
        )

    def write_summary(self, summary: dict[str, Any]) -> None:
        _atomic_write_json(self.run_dir / "summary.json", summary)

    def last_completed_round(self) -> int:
        """Highest round number with a checkpoint marked complete, or 0 if none.

        Checkpoint files are named ``round_<N>.pt`` and are only renamed into place from a
        ``.tmp`` suffix after a fully successful write (see checkpointing in M4/M5), so presence
        of ``round_<N>.pt`` is itself the "phase-consistent, completed round" marker.
        """
        best = 0
        for p in self.checkpoints_dir.glob("round_*.pt"):
            try:
                n = int(p.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            best = max(best, n)
        return best

    def checkpoint_due(self, server_round: int) -> bool:
        return (
            server_round % self.config.checkpoint_interval == 0
            or server_round in self.config.checkpoint_rounds
            or server_round == self.config.num_server_rounds
        )

    def save_server_checkpoint(
        self,
        server_round: int,
        arrays: dict[str, Any],
        server_model_state: dict[str, Any] | None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Atomically persist a phase-consistent completed server round."""
        path = self.checkpoints_dir / f"round_{server_round}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(
            {
                "round": server_round,
                "arrays": arrays,
                "server_model_state": server_model_state,
                "extra": extra or {},
            },
            tmp,
        )
        tmp.replace(path)
        return path

    def load_latest_server_checkpoint(self) -> dict[str, Any] | None:
        server_round = self.last_completed_round()
        if server_round == 0:
            return None
        return torch.load(
            self.checkpoints_dir / f"round_{server_round}.pt",
            map_location="cpu",
            weights_only=False,
        )
