"""Experiment matrix runner (M7 "Paper Experiments"): drives many ``flwr run`` invocations from one
declarative YAML matrix file (``configs/experiments*.yaml``) -- the main algorithm/scenario matrix,
SSFL ablations, threshold study, label-representation study, and multi-seed robustness sweeps.

Each matrix entry names a base ``configs/<profile>.yaml`` plus a dict of ``ExperimentConfig`` field
overrides. Flower's ``--run-config`` can only override the small allowlist declared in
``pyproject.toml``'s ``[tool.flwr.app.config]`` (``profile``/``algorithm``/``scenario``/``device``;
see REPRODUCIBILITY.md #18), so the runner instead resolves+validates the merged config in-process,
writes it out as its own generated profile YAML under ``artifacts/generated_configs/`` (a fallback
``experiment_config_from_run_config`` checks after ``configs/``), and selects it via
``profile=<entry name>``.

Resumable: an entry whose deterministic run directory already has ``summary.json`` is skipped under
``--resume`` rather than re-executed, matching ``RunContext``'s own run-id/run-dir determinism.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from ssfl.config import ExperimentConfig, compute_run_id, load_yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
GENERATED_DIR = REPO_ROOT / "artifacts" / "generated_configs"


def _normalize_key(key: str) -> str:
    return key.replace("-", "_")


def build_matrix_configs(
    matrix_path: Path, configs_dir: Path = CONFIGS_DIR
) -> list[tuple[str, ExperimentConfig]]:
    """Resolve a matrix YAML (``entries: [{name, base_profile, overrides}, ...]``) into validated
    ``(name, ExperimentConfig)`` pairs. Fails fast on any bad entry before any run starts."""
    spec = load_yaml(matrix_path)
    resolved: list[tuple[str, ExperimentConfig]] = []
    seen_names: set[str] = set()
    for entry in spec["entries"]:
        name = entry["name"]
        if name in seen_names:
            raise ValueError(f"duplicate matrix entry name {name!r} in {matrix_path}")
        if (configs_dir / f"{name}.yaml").exists():
            raise ValueError(f"matrix entry name {name!r} collides with configs/{name}.yaml")
        seen_names.add(name)
        base = load_yaml(configs_dir / f"{entry['base_profile']}.yaml")
        base = {_normalize_key(k): v for k, v in base.items()}
        overrides = {_normalize_key(k): v for k, v in entry.get("overrides", {}).items()}
        merged = {**base, **overrides, "profile": name}
        resolved.append((name, ExperimentConfig.model_validate(merged)))
    return resolved


def _dataset_manifest_hash(config: ExperimentConfig) -> str | None:
    manifest_path = config.data_path / "dataset_manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text()).get("manifest_hash")


def _run_dir(config: ExperimentConfig) -> Path:
    run_id = compute_run_id(config, dataset_manifest_hash=_dataset_manifest_hash(config))
    return config.output_path / run_id


def _federation_config(config: ExperimentConfig) -> str:
    """Flower/Ray resources, including explicit CUDA visibility for ClientApp actors."""
    fields = [
        f"num-supernodes={config.num_clients()}",
        f"client-resources-num-cpus={int(config.client_num_cpus)}",
        f"client-resources-num-gpus={config.client_num_gpus}",
        f"init-args-num-cpus={max(1, int(config.max_concurrent_clients * config.client_num_cpus))}",
    ]
    if config.client_num_gpus > 0:
        fields.append("init-args-num-gpus=1")
    return " ".join(fields)


def compress_completed_jsonl(run_dir: Path) -> list[Path]:
    """Losslessly compress completed attempt logs, leaving active runs untouched."""
    compressed: list[Path] = []
    for source in sorted((run_dir / "attempts").glob("**/*.jsonl")):
        target = source.with_suffix(f"{source.suffix}.gz")
        temporary = target.with_suffix(f"{target.suffix}.tmp")
        with source.open("rb") as input_stream, temporary.open("wb") as raw_output:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_output, compresslevel=6, mtime=0
            ) as output_stream:
                shutil.copyfileobj(input_stream, output_stream)
        temporary.replace(target)
        source.unlink()
        compressed.append(target)
    return compressed


def run_suite(
    matrix_path: Path,
    resume: bool,
    dry_run: bool = False,
    configs_dir: Path = CONFIGS_DIR,
    generated_dir: Path = GENERATED_DIR,
    report_dir: Path = REPO_ROOT / "artifacts" / "runs",
) -> int:
    generated_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    failed = 0

    for name, config in build_matrix_configs(matrix_path, configs_dir=configs_dir):
        run_dir = _run_dir(config)
        summary_path = run_dir / "summary.json"
        if resume and summary_path.exists():
            compressed = compress_completed_jsonl(run_dir)
            if compressed:
                print(f"[run_suite] COMPRESS {name} ({len(compressed)} JSONL files)")
            print(f"[run_suite] SKIP  {name} (already complete: {summary_path})")
            results.append({"name": name, "status": "skipped", "run_dir": str(run_dir)})
            continue

        generated_path = generated_dir / f"{name}.yaml"
        generated_path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True))

        cmd = [
            "flwr",
            "run",
            str(REPO_ROOT),
            "--run-config",
            # flwr's --run-config value is TOML: string values must be quoted or it rejects the
            # whole string as "invalid format" (int values like scenario must NOT be quoted).
            f'profile="{name}" algorithm="{config.algorithm.value}" scenario={config.scenario.value} '
            f'device="{config.device.value}"',
            "--federation-config",
            # no "options." prefix (deprecated/rejected by the installed flwr 1.32.1).
            _federation_config(config),
            # without --stream, `flwr run` submits to the SuperLink and returns immediately
            # instead of waiting for the run to finish -- --stream is what makes this subprocess
            # call actually block until the run completes (confirmed live: omitting it returned
            # in <1s with no training having happened at all).
            "--stream",
        ]
        print(f"[run_suite] RUN   {name}: {' '.join(cmd)}")
        if dry_run:
            results.append({"name": name, "status": "dry_run", "run_dir": str(run_dir)})
            continue

        proc = subprocess.run(cmd, cwd=REPO_ROOT)
        if proc.returncode != 0 or not summary_path.exists():
            print(f"[run_suite] FAIL  {name} (exit={proc.returncode})")
            results.append(
                {
                    "name": name,
                    "status": "failed",
                    "run_dir": str(run_dir),
                    "exit_code": proc.returncode,
                }
            )
            failed += 1
        else:
            compressed = compress_completed_jsonl(run_dir)
            print(f"[run_suite] COMPRESS {name} ({len(compressed)} JSONL files)")
            print(f"[run_suite] DONE  {name}")
            results.append({"name": name, "status": "done", "run_dir": str(run_dir)})

    report_path = report_dir / f"suite_report_{matrix_path.stem}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"matrix": str(matrix_path), "results": results}, indent=2))
    print(f"[run_suite] report written to {report_path}")
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a declarative SSFL experiment matrix.")
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument(
        "--resume", action="store_true", help="skip entries whose run already completed"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="resolve+write configs without launching flwr run"
    )
    args = parser.parse_args()
    sys.exit(run_suite(args.matrix, resume=args.resume, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
