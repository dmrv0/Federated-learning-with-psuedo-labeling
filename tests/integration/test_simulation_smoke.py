"""Real ``flwr run`` simulation smoke tests -- each spawns an actual Ray-backed simulation and
takes minutes, not seconds, so all of these are ``@pytest.mark.slow`` and excluded from the default
``pytest -q`` run (see pyproject.toml's ``-m "not slow"`` addopts). Run explicitly with
``pytest -m slow tests/integration``.

Runs against the repo's already-prepared ``artifacts/data`` (the same tree every manual smoke
verification in REPRODUCIBILITY.md used), not a throwaway synthetic tree -- these tests assert the
same "does a real 2-round simulation complete and produce sane artifacts" property the plan's M9
"Integration" bullet wants, for all four algorithms uniformly.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ssfl.config import compute_run_id, experiment_config_from_run_config
from ssfl.experiments.run_suite import REPO_ROOT, _dataset_manifest_hash

ALGORITHMS = ["ssfl", "fl", "fd", "dsfl"]


def _run_dir_for(algorithm: str, scenario: int = 1) -> Path:
    config = experiment_config_from_run_config(
        {"profile": "smoke", "algorithm": algorithm, "scenario": scenario, "device": "cpu"}
    )
    run_id = compute_run_id(config, dataset_manifest_hash=_dataset_manifest_hash(config))
    return config.output_path / run_id


def _run_flwr(
    algorithm: str, scenario: int = 1, num_supernodes: int = 27
) -> subprocess.CompletedProcess:
    cmd = [
        "flwr",
        "run",
        str(REPO_ROOT),
        "--run-config",
        f'profile="smoke" algorithm="{algorithm}" scenario={scenario} device="cpu"',
        "--federation-config",
        f"num-supernodes={num_supernodes}",
        "--stream",
    ]
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800)


@pytest.mark.slow
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_two_round_smoke_completes(algorithm: str) -> None:
    proc = _run_flwr(algorithm)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]

    run_dir = _run_dir_for(algorithm)
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["algorithm"] == algorithm
    assert summary["final_round"] == 2

    attempt = json.loads((run_dir / "current_attempt.json").read_text())
    attempt_dir = Path(attempt["attempt_dir"])
    events = [json.loads(line) for line in (attempt_dir / "events.jsonl").read_text().splitlines()]
    assert events[0]["message"] == "run_start"
    assert events[-1]["message"] == "run_end"
    assert any(e["message"] == "aggregate" and e.get("phase") == "train" for e in events)
    assert any(e["message"] == "aggregate" and e.get("phase") == "evaluate" for e in events)

    assert (run_dir / "communication.parquet").exists()
    assert (run_dir / "metrics.parquet").exists()
    assert (run_dir / "checkpoints" / "round_2.pt").exists()
    assert (attempt_dir / "telemetry" / "server.jsonl").exists()
    assert len(list((attempt_dir / "telemetry" / "clients").glob("*.jsonl"))) == 27


@pytest.mark.slow
def test_identical_seed_runs_are_deterministic() -> None:
    """Two smoke runs of the same algorithm/scenario/seed must agree on final metrics -- rerunning
    is the whole point of an SSFL run directory's determinism guarantee (REPRODUCIBILITY.md).
    Both runs resolve to the same deterministic run_id/run_dir, so the first run's summary is
    captured before the second run overwrites it in place.

    Near-equality, not bit-exact: see REPRODUCIBILITY.md #28. This build machine is an Apple M4
    (heterogeneous P/E cores); Ray doesn't pin worker-process CPU affinity, and Apple's Accelerate/
    NEON matmul kernels produce different low-order-bit floats between core types for bit-identical
    input. That ~1-ULP noise gets amplified 10x by DS-FL's sharpen(T=0.1) and compounds over rounds.
    Two isolation probes (raw-PyTorch training and the real client_predict_step/train_supervised
    path against real data) were both bit-identical outside Ray, including under 6-way concurrent
    process contention -- confirming this is Ray-placement/hardware-level, not an application bug,
    and not reachable from a Python-level fix."""
    run_dir = _run_dir_for("dsfl")

    first = _run_flwr("dsfl")
    assert first.returncode == 0, first.stdout[-4000:] + first.stderr[-4000:]
    first_summary = json.loads((run_dir / "summary.json").read_text())

    second = _run_flwr("dsfl")
    assert second.returncode == 0, second.stdout[-4000:] + second.stderr[-4000:]
    second_summary = json.loads((run_dir / "summary.json").read_text())

    first_metrics = first_summary["final_centralized_metrics"]
    second_metrics = second_summary["final_centralized_metrics"]
    assert first_metrics.keys() == second_metrics.keys()
    for key, first_value in first_metrics.items():
        second_value = second_metrics[key]
        if key == "loss":
            assert first_value == pytest.approx(second_value, rel=0.05)
        else:
            assert first_value == pytest.approx(second_value, abs=0.05)
