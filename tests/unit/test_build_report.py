import json

import pandas as pd
import yaml

from ssfl.reporting.build_report import (
    _ablation_label,
    _comm_at_accuracy,
    _cumulative_comm_mb,
    _cumulative_paper_comm_mb,
    _label_study_label,
    _threshold_label,
    build_report,
    build_table_ii,
    build_table_iii,
    build_table_iv,
    load_runs,
)

BASE_CONFIG = {
    "profile": "main",
    "algorithm": "ssfl",
    "backbone": "cnn",
    "scenario": 1,
    "ssfl_discriminator_mode": "enabled",
    "ssfl_voting_mode": "enabled",
    "ssfl_threshold_policy": "median",
    "ssfl_label_representation": "hard",
    "ssfl_soft_label_round_decimals": None,
}


def _make_run(tmp_path, run_id, config_overrides, metrics_rows, comms_rows=None, final_round=None):
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    config = {**BASE_CONFIG, **config_overrides}
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    (run_dir / "summary.json").write_text(
        json.dumps(
            {"final_round": final_round if final_round is not None else metrics_rows[-1]["round"]}
        )
    )
    pd.DataFrame(metrics_rows).to_parquet(run_dir / "metrics.parquet", index=False)
    if comms_rows:
        pd.DataFrame(comms_rows).to_parquet(run_dir / "communication.parquet", index=False)
    return run_dir


def _comm_row(round_n, logical_bytes, *, paper_bytes=None, direction="client_to_server"):
    return {
        "algorithm": "ssfl",
        "scenario": 1,
        "round": round_n,
        "phase": "train",
        "direction": direction,
        "logical_bytes": logical_bytes,
        "paper_bytes": logical_bytes if paper_bytes is None else paper_bytes,
        "serialized_bytes": logical_bytes + 10,
    }


def test_cumulative_comm_mb_sums_and_accumulates_across_rounds():
    comms = pd.DataFrame(
        [_comm_row(1, 1024 * 1024), _comm_row(1, 1024 * 1024), _comm_row(2, 1024 * 1024)]
    )
    cum = _cumulative_comm_mb(comms)
    assert cum[1] == 2.0
    assert cum[2] == 3.0


def test_paper_comm_uses_mean_client_uplink_and_ignores_downlink():
    comms = pd.DataFrame(
        [
            _comm_row(1, 999, paper_bytes=1024 * 1024),
            _comm_row(1, 999, paper_bytes=3 * 1024 * 1024),
            _comm_row(1, 9 * 1024 * 1024, direction="server_to_client"),
            _comm_row(2, 999, paper_bytes=2 * 1024 * 1024),
        ]
    )
    cum = _cumulative_paper_comm_mb(comms)
    assert cum[1] == 2.0
    assert cum[2] == 4.0


def test_comm_at_accuracy_returns_first_round_crossing_threshold():
    metrics = pd.DataFrame(
        [
            {"round": 1, "accuracy": 0.3},
            {"round": 2, "accuracy": 0.6},
            {"round": 3, "accuracy": 0.9},
        ]
    )
    cum_mb = pd.Series({1: 1.0, 2: 2.0, 3: 3.0})
    assert _comm_at_accuracy(metrics, cum_mb, 0.5) == 2.0
    assert _comm_at_accuracy(metrics, cum_mb, 0.99) is None


def test_ablation_label_covers_all_five_combinations():
    assert (
        _ablation_label({"ssfl_discriminator_mode": "enabled", "ssfl_voting_mode": "enabled"})
        == "Ours"
    )
    assert (
        _ablation_label({"ssfl_discriminator_mode": "disabled", "ssfl_voting_mode": "enabled"})
        == "Ours w/o Discriminating"
    )
    assert (
        _ablation_label({"ssfl_discriminator_mode": "enabled", "ssfl_voting_mode": "disabled"})
        == "Ours w/o Voting"
    )
    assert (
        _ablation_label({"ssfl_discriminator_mode": "disabled", "ssfl_voting_mode": "disabled"})
        == "Ours w/o Discriminating and Voting"
    )
    assert (
        _ablation_label({"ssfl_discriminator_mode": "simple_filter", "ssfl_voting_mode": "enabled"})
        == "Simply Filtering"
    )


def test_threshold_label_formats_fixed_and_median():
    assert _threshold_label({"ssfl_threshold_policy": "median"}) == "Confidence Threshold Median"
    assert _threshold_label({"ssfl_threshold_policy": "fixed_0_8"}) == "Confidence Threshold 0.8"


def test_label_study_label_hard_vs_soft():
    assert _label_study_label({"ssfl_label_representation": "hard"}) == "Hard Label"
    assert (
        _label_study_label(
            {"ssfl_label_representation": "soft", "ssfl_soft_label_round_decimals": 4}
        )
        == "Soft Label w. 4f"
    )


def test_build_table_ii_populates_matched_cell_and_leaves_others_blank(tmp_path):
    _make_run(
        tmp_path,
        "ssfl-run",
        {"profile": "ssfl_cnn_s1", "algorithm": "ssfl", "backbone": "cnn", "scenario": 1},
        [{"round": 1, "accuracy": 0.5, "macro_f1": 0.4, "macro_precision": 0.6}],
    )
    records = load_runs(tmp_path)
    table = build_table_ii(records)
    ours_s1 = table[(table["method"] == "Ours") & (table["scenario"] == 1)].iloc[0]
    assert ours_s1["repro_accuracy"] == 0.5
    assert ours_s1["paper_accuracy"] == 0.8740
    fl_s1 = table[(table["method"] == "FL") & (table["scenario"] == 1)].iloc[0]
    assert pd.isna(fl_s1["repro_accuracy"])
    assert fl_s1["paper_accuracy"] == 0.8611


def test_main_matrix_prefers_run_with_more_completed_rounds(tmp_path):
    _make_run(
        tmp_path,
        "ssfl-smoke",
        {"profile": "smoke", "algorithm": "ssfl", "backbone": "cnn", "scenario": 1},
        [{"round": 2, "accuracy": 0.1, "macro_f1": 0.1, "macro_precision": 0.1}],
        final_round=2,
    )
    _make_run(
        tmp_path,
        "ssfl-paper",
        {"profile": "ssfl_cnn_s1", "algorithm": "ssfl", "backbone": "cnn", "scenario": 1},
        [{"round": 200, "accuracy": 0.87, "macro_f1": 0.86, "macro_precision": 0.92}],
        final_round=200,
    )
    records = load_runs(tmp_path)
    table = build_table_ii(records)
    ours_s1 = table[(table["method"] == "Ours") & (table["scenario"] == 1)].iloc[0]
    assert ours_s1["run_id"] == "ssfl-paper"
    assert ours_s1["repro_accuracy"] == 0.87


def test_build_table_iii_matches_exact_round_only(tmp_path):
    _make_run(
        tmp_path,
        "ssfl-run",
        {"profile": "ssfl_cnn_s1", "algorithm": "ssfl", "backbone": "cnn", "scenario": 1},
        [{"round": 10, "accuracy": 0.7, "macro_f1": 0.6, "macro_precision": 0.7}],
        final_round=10,
    )
    records = load_runs(tmp_path)
    table = build_table_iii(records)
    ours_s1 = table[(table["method"] == "Ours") & (table["scenario"] == 1)].iloc[0]
    assert ours_s1["repro_r10"] == 70.0
    assert pd.isna(ours_s1["repro_r50"])


def test_build_table_iv_computes_c50_c75_and_top_acc(tmp_path):
    metrics_rows = [
        {"round": 1, "accuracy": 0.3, "macro_f1": 0.2, "macro_precision": 0.3},
        {"round": 2, "accuracy": 0.6, "macro_f1": 0.5, "macro_precision": 0.6},
        {"round": 3, "accuracy": 0.9, "macro_f1": 0.8, "macro_precision": 0.9},
    ]
    comms_rows = [_comm_row(1, 1024 * 1024), _comm_row(2, 1024 * 1024), _comm_row(3, 1024 * 1024)]
    _make_run(
        tmp_path,
        "ssfl-run",
        {"profile": "ssfl_cnn_s1", "algorithm": "ssfl", "backbone": "cnn", "scenario": 1},
        metrics_rows,
        comms_rows,
        final_round=3,
    )
    records = load_runs(tmp_path)
    table = build_table_iv(records)
    ours_s1 = table[(table["method"] == "Ours") & (table["scenario"] == 1)].iloc[0]
    assert ours_s1["repro_c50_mb"] == 2.0
    assert ours_s1["repro_c75_mb"] == 3.0
    assert ours_s1["repro_top_acc"] == 0.9
    assert ours_s1["repro_c_top_acc_mb"] == 3.0


def test_build_report_end_to_end_writes_tables_and_report(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    metrics_rows = [{"round": 1, "accuracy": 0.5, "macro_f1": 0.4, "macro_precision": 0.5}]
    _make_run(
        runs_dir,
        "ssfl-run",
        {"profile": "ssfl_cnn_s1", "algorithm": "ssfl", "backbone": "cnn", "scenario": 1},
        metrics_rows,
    )

    output_dir = tmp_path / "report"
    build_report(runs_dir, output_dir, data_dir=tmp_path / "no_such_data_dir")

    assert (output_dir / "table_ii.csv").exists()
    assert (output_dir / "table_iii.parquet").exists()
    assert (output_dir / "table_iv.csv").exists()
    assert (output_dir / "report.md").exists()
    assert "Table II" in (output_dir / "report.md").read_text()
