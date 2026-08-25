from pathlib import Path

import pytest
from pydantic import ValidationError

from ssfl.config import (
    ExperimentConfig,
    LabelRepresentation,
    VotingMode,
    experiment_config_from_run_config,
    load_experiment_config,
    load_yaml,
    parse_run_config_string,
)

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
ALL_PROFILES = ["paper", "paper_batch100", "smoke", "robustness", "deployment", "debug"]


@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_all_profiles_validate(profile: str) -> None:
    cfg = load_experiment_config(CONFIGS_DIR / f"{profile}.yaml")
    assert cfg.profile == profile


def test_unknown_key_rejected() -> None:
    base = load_yaml(CONFIGS_DIR / "paper.yaml")
    base["totally_unknown_field"] = 1
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(base)


@pytest.mark.parametrize(
    "label_repr,voting",
    [
        (LabelRepresentation.hard, VotingMode.disabled),
        (LabelRepresentation.soft, VotingMode.enabled),
    ],
)
def test_invalid_label_voting_combination_rejected(label_repr, voting) -> None:
    base = load_yaml(CONFIGS_DIR / "paper.yaml")
    base["ssfl_label_representation"] = label_repr.value
    base["ssfl_voting_mode"] = voting.value
    if label_repr == LabelRepresentation.soft:
        base["ssfl_soft_label_round_decimals"] = 4
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(base)


def test_soft_label_requires_valid_rounding() -> None:
    base = load_yaml(CONFIGS_DIR / "paper.yaml")
    base["ssfl_label_representation"] = "soft"
    base["ssfl_voting_mode"] = "disabled"
    base["ssfl_soft_label_round_decimals"] = 3  # not in {2,4,6,8}
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(base)


def test_bad_scenario_rejected() -> None:
    base = load_yaml(CONFIGS_DIR / "paper.yaml")
    base["scenario"] = 4
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(base)


def test_run_config_string_overrides_profile() -> None:
    overrides = parse_run_config_string("algorithm=fl scenario=2 num-server-rounds=5 seed=99")
    cfg = load_experiment_config(CONFIGS_DIR / "smoke.yaml", overrides=overrides)
    assert cfg.algorithm.value == "fl"
    assert cfg.scenario.value == 2
    assert cfg.num_server_rounds == 5
    assert cfg.seed == 99


def test_empty_resume_default_is_treated_as_none() -> None:
    cfg = experiment_config_from_run_config(
        {
            "profile": "smoke",
            "algorithm": "ssfl",
            "scenario": 1,
            "device": "cpu",
            "resume-from": "",
        }
    )
    assert cfg.resume_from is None
    assert cfg.data_path == CONFIGS_DIR.parent / "artifacts/data"
    assert cfg.output_path == CONFIGS_DIR.parent / "artifacts/runs"


def test_num_clients_matches_scenario() -> None:
    base = load_yaml(CONFIGS_DIR / "paper.yaml")
    for scenario, expected in [(1, 27), (2, 89), (3, 89)]:
        base["scenario"] = scenario
        cfg = ExperimentConfig.model_validate(base)
        assert cfg.num_clients() == expected


def test_config_hash_stable_and_excludes_paths() -> None:
    base = load_yaml(CONFIGS_DIR / "paper.yaml")
    cfg_a = ExperimentConfig.model_validate(base)
    base2 = dict(base)
    base2["output_path"] = "/some/other/path"
    cfg_b = ExperimentConfig.model_validate(base2)
    assert cfg_a.config_hash() == cfg_b.config_hash()


def test_paper_text_batch_overrides_effective_batch_size() -> None:
    cfg = load_experiment_config(CONFIGS_DIR / "paper.yaml")
    assert cfg.effective_batch_size == 80
    cfg100 = load_experiment_config(CONFIGS_DIR / "paper_batch100.yaml")
    assert cfg100.effective_batch_size == 100


def test_backbone_restricted_to_ssfl() -> None:
    base = load_yaml(CONFIGS_DIR / "paper.yaml")
    base["algorithm"] = "fl"
    base["backbone"] = "mlp"
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(base)
