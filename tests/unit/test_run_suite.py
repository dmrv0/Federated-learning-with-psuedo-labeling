import gzip
import json
import subprocess

import pytest
import yaml

from ssfl.config import ExperimentConfig
from ssfl.experiments.run_suite import (
    _federation_config,
    _run_dir,
    build_matrix_configs,
    compress_completed_jsonl,
    run_suite,
)


def _write_yaml(path, data) -> None:
    path.write_text(yaml.safe_dump(data))


@pytest.fixture
def configs_dir(tmp_path):
    d = tmp_path / "configs"
    d.mkdir()
    _write_yaml(
        d / "base.yaml",
        {
            "profile": "base",
            "run_kind": "extension",
            "algorithm": "ssfl",
            "backbone": "cnn",
            "scenario": 1,
            "seed": 2023,
            "data_path": str(tmp_path / "data"),
            "output_path": str(tmp_path / "runs"),
            "num_server_rounds": 2,
            "local_epochs": 1,
            "batch_size": 16,
            "device": "cpu",
            "ssfl_threshold_policy": "median",
            "ssfl_discriminator_mode": "enabled",
            "ssfl_voting_mode": "enabled",
            "ssfl_label_representation": "hard",
        },
    )
    return d


def test_build_matrix_configs_resolves_overrides(tmp_path, configs_dir) -> None:
    matrix_path = tmp_path / "matrix.yaml"
    _write_yaml(
        matrix_path,
        {
            "entries": [
                {
                    "name": "variant_a",
                    "base_profile": "base",
                    "overrides": {"ssfl_discriminator_mode": "disabled"},
                }
            ]
        },
    )
    resolved = build_matrix_configs(matrix_path, configs_dir=configs_dir)
    assert len(resolved) == 1
    name, config = resolved[0]
    assert name == "variant_a"
    assert config.profile == "variant_a"  # entry name becomes the generated profile identity
    assert config.ssfl_discriminator_mode.value == "disabled"


def test_federation_config_allocates_gpu_fraction_and_concurrency(configs_dir) -> None:
    config = ExperimentConfig.model_validate(
        {
            **yaml.safe_load((configs_dir / "base.yaml").read_text()),
            "device": "cuda",
            "client_num_gpus": 0.125,
            "max_concurrent_clients": 8,
        }
    )
    rendered = _federation_config(config)
    assert "client-resources-num-cpus=1 " in rendered
    assert "client-resources-num-cpus=1.0" not in rendered
    assert "client-resources-num-gpus=0.125" in rendered
    assert "init-args-num-gpus=1" in rendered
    assert "num-supernodes=27" in rendered


def test_build_matrix_configs_rejects_duplicate_names(tmp_path, configs_dir) -> None:
    matrix_path = tmp_path / "matrix.yaml"
    _write_yaml(
        matrix_path,
        {
            "entries": [
                {"name": "dup", "base_profile": "base"},
                {"name": "dup", "base_profile": "base"},
            ]
        },
    )
    with pytest.raises(ValueError, match="duplicate"):
        build_matrix_configs(matrix_path, configs_dir=configs_dir)


def test_build_matrix_configs_rejects_name_colliding_with_existing_profile(
    tmp_path, configs_dir
) -> None:
    matrix_path = tmp_path / "matrix.yaml"
    _write_yaml(matrix_path, {"entries": [{"name": "base", "base_profile": "base"}]})
    with pytest.raises(ValueError, match="collides"):
        build_matrix_configs(matrix_path, configs_dir=configs_dir)


def test_build_matrix_configs_fails_fast_on_invalid_ssfl_combination(tmp_path, configs_dir) -> None:
    matrix_path = tmp_path / "matrix.yaml"
    _write_yaml(
        matrix_path,
        # hard label representation + voting disabled is not one of the five valid combinations.
        {
            "entries": [
                {
                    "name": "bad",
                    "base_profile": "base",
                    "overrides": {"ssfl_voting_mode": "disabled"},
                }
            ]
        },
    )
    with pytest.raises(Exception, match="invalid ssfl_label_representation"):
        build_matrix_configs(matrix_path, configs_dir=configs_dir)


def test_run_suite_dry_run_writes_generated_config_and_report_without_launching(
    tmp_path, configs_dir, monkeypatch
) -> None:
    matrix_path = tmp_path / "matrix.yaml"
    _write_yaml(matrix_path, {"entries": [{"name": "variant_a", "base_profile": "base"}]})
    monkeypatch.setattr("ssfl.config._git_commit", lambda: "testcommit")

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("subprocess.run must not be called in dry_run mode")

    monkeypatch.setattr(subprocess, "run", _fail_if_called)

    generated_dir = tmp_path / "generated"
    report_dir = tmp_path / "runs_out"
    exit_code = run_suite(
        matrix_path,
        resume=False,
        dry_run=True,
        configs_dir=configs_dir,
        generated_dir=generated_dir,
        report_dir=report_dir,
    )

    assert exit_code == 0
    assert (generated_dir / "variant_a.yaml").exists()
    report = json.loads((report_dir / "suite_report_matrix.json").read_text())
    assert report["results"] == [
        {"name": "variant_a", "status": "dry_run", "run_dir": report["results"][0]["run_dir"]}
    ]


def test_run_suite_resume_skips_entry_with_existing_summary(
    tmp_path, configs_dir, monkeypatch
) -> None:
    matrix_path = tmp_path / "matrix.yaml"
    _write_yaml(matrix_path, {"entries": [{"name": "variant_a", "base_profile": "base"}]})
    # Fixed stub so the run_dir computed here (pre-patch) matches the one run_suite computes
    # internally (post-patch) -- both must agree for the resume-skip check to mean anything.
    monkeypatch.setattr("ssfl.config._git_commit", lambda: "testcommit")

    [(_, config)] = build_matrix_configs(matrix_path, configs_dir=configs_dir)
    run_dir = _run_dir(config)
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text("{}")
    telemetry = run_dir / "attempts" / "attempt-1" / "telemetry" / "server.jsonl"
    telemetry.parent.mkdir(parents=True)
    telemetry.write_text('{"event":"run_end"}\n')

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("subprocess.run must not be called for an already-completed entry")

    monkeypatch.setattr(subprocess, "run", _fail_if_called)

    generated_dir = tmp_path / "generated"
    report_dir = tmp_path / "runs_out"
    exit_code = run_suite(
        matrix_path,
        resume=True,
        dry_run=False,
        configs_dir=configs_dir,
        generated_dir=generated_dir,
        report_dir=report_dir,
    )

    assert exit_code == 0
    assert not (generated_dir / "variant_a.yaml").exists()  # skipped before ever writing a profile
    assert not telemetry.exists()
    with gzip.open(telemetry.with_suffix(".jsonl.gz"), "rt") as stream:
        assert stream.read() == '{"event":"run_end"}\n'
    report = json.loads((report_dir / "suite_report_matrix.json").read_text())
    assert report["results"][0]["status"] == "skipped"


def test_compress_completed_jsonl_is_lossless_and_idempotent(tmp_path) -> None:
    first = tmp_path / "attempts" / "one" / "events.jsonl"
    second = tmp_path / "attempts" / "one" / "telemetry" / "clients" / "c1.jsonl"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text('{"event":"round_end","round":1}\n')
    second.write_text('{"event":"training_epoch","epoch":1}\n')

    compressed = compress_completed_jsonl(tmp_path)

    assert compressed == [first.with_suffix(".jsonl.gz"), second.with_suffix(".jsonl.gz")]
    assert not first.exists()
    assert not second.exists()
    with gzip.open(compressed[0], "rt") as stream:
        assert stream.read() == '{"event":"round_end","round":1}\n'
    with gzip.open(compressed[1], "rt") as stream:
        assert stream.read() == '{"event":"training_epoch","epoch":1}\n'
    assert compress_completed_jsonl(tmp_path) == []
