import json

import torch

from ssfl.config import DeviceKind, capture_environment_snapshot, load_experiment_config
from ssfl.device import resolve_device
from ssfl.logging_utils import ForbiddenLogFieldError, bind, configure_logging
from ssfl.run_context import RunContext, prune_superseded_checkpoints
from ssfl.seeding import configure_determinism, make_generator, seed_everything

import pytest
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


def test_seed_everything_reproducible() -> None:
    seed_everything(123)
    a = torch.randn(5)
    seed_everything(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_make_generator_is_seeded_and_reproducible() -> None:
    g1 = make_generator(42)
    x1 = torch.randn(5, generator=g1)
    g2 = make_generator(42)
    x2 = torch.randn(5, generator=g2)
    assert torch.equal(x1, x2)


def test_configure_determinism_enabled_no_crash() -> None:
    warnings = configure_determinism(True)
    assert isinstance(warnings, list)
    torch.use_deterministic_algorithms(False)  # reset for other tests in the session


def test_resolve_device_cpu_explicit() -> None:
    assert resolve_device(DeviceKind.cpu, deterministic=True) == torch.device("cpu")


def test_resolve_device_auto_deterministic_avoids_mps() -> None:
    device = resolve_device(DeviceKind.auto, deterministic=True)
    assert device.type in ("cpu", "cuda")


def test_environment_snapshot_has_expected_fields() -> None:
    snap = capture_environment_snapshot()
    assert "python_version" in snap
    assert "torch_version" in snap
    assert "dependency_versions" in snap
    assert snap["dependency_versions"]["flwr"] is not None


def test_logging_forbidden_field_rejected() -> None:
    logger = configure_logging()
    with pytest.raises(ForbiddenLogFieldError):
        bind(logger, private_features=[1, 2, 3])


def test_logging_allows_normal_fields(caplog) -> None:
    logger = configure_logging()
    adapter = bind(logger, run_id="r1", algorithm="ssfl", round=1)
    adapter.info("round complete")


def test_run_context_create_and_resume(tmp_path) -> None:
    cfg = load_experiment_config(CONFIGS_DIR / "smoke.yaml")
    cfg = cfg.model_copy(update={"output_path": tmp_path})
    ctx = RunContext.create(cfg)

    assert (ctx.run_dir / "resolved_config.yaml").exists()
    assert (ctx.run_dir / "environment.json").exists()
    assert (ctx.run_dir / "code_version.json").exists()
    assert ctx.checkpoints_dir.exists()
    assert ctx.plots_dir.exists()

    env = json.loads((ctx.run_dir / "environment.json").read_text())
    assert "torch_version" in env

    resumed = RunContext.resume(ctx.run_dir)
    assert resumed.run_id == ctx.run_id
    assert resumed.config.algorithm == cfg.algorithm


def test_run_context_deterministic_run_id(tmp_path) -> None:
    cfg = load_experiment_config(CONFIGS_DIR / "smoke.yaml")
    cfg_a = cfg.model_copy(update={"output_path": tmp_path / "a"})
    cfg_b = cfg.model_copy(update={"output_path": tmp_path / "b"})
    ctx_a = RunContext.create(cfg_a)
    ctx_b = RunContext.create(cfg_b)
    assert ctx_a.run_id == ctx_b.run_id  # same config/seed -> same run id regardless of output path


def test_checkpoint_retention_keeps_milestones_and_latest(tmp_path) -> None:
    for round_number in range(1, 7):
        (tmp_path / f"round_{round_number}.pt").write_bytes(b"checkpoint")

    pruned = prune_superseded_checkpoints(
        tmp_path,
        current_round=6,
        pinned_rounds=(2, 5, 10),
    )

    assert {path.name for path in pruned} == {
        "round_1.pt",
        "round_3.pt",
        "round_4.pt",
    }
    assert {path.name for path in tmp_path.glob("round_*.pt")} == {
        "round_2.pt",
        "round_5.pt",
        "round_6.pt",
    }
