"""M8/M10 reporting: paper-style Tables II-IV and Figures 2-6, built entirely from existing
``artifacts/runs/*/{summary.json,resolved_config.yaml,metrics.parquet,communication.parquet,
confusion_matrices.npz}`` and ``artifacts/data/scenarios/*_allocation_stats.json`` -- no new
run-time instrumentation needed, and it degrades gracefully to whatever runs exist (CPU
smoke-scale today; paper-scale 200-round runs once available, per REPRODUCIBILITY.md's
GPU-gated verification decision).

Paper reference values (``PAPER_TABLE_*``) are transcribed directly from Zhao et al., Tables
II-IV and Figs 2-6 (IEEE Internet of Things Journal, Vol. 10, No. 10, 15 May 2023, pp. 8651-8655).

CLI: ``python -m ssfl.reporting.build_report --runs artifacts/runs --output artifacts/report``
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

CLASS_NAMES = [
    "Ben",
    "G_Co",
    "G_Ju",
    "G_Sc",
    "G_TCP",
    "G_UDP",
    "M_Ack",
    "M_Sc",
    "M_Syn",
    "M_UDP",
    "M_UDPp",
]

METHOD_LABELS = {
    ("fl", "cnn"): "FL",
    ("fd", "cnn"): "FD",
    ("dsfl", "cnn"): "DS-FL",
    ("ssfl", "cnn"): "Ours",
    ("ssfl", "mlp"): "MLP",
    ("ssfl", "lstm"): "LSTM",
}

# --- Paper reference values (Table II, accuracy/F1/precision as fractions) -----------------------
PAPER_TABLE_II = {
    ("FL", 1): {"accuracy": 0.8611, "f1": 0.8513, "precision": 0.9129},
    ("FD", 1): {"accuracy": 0.4854, "f1": 0.3576, "precision": 0.3369},
    ("DS-FL", 1): {"accuracy": 0.5049, "f1": 0.4085, "precision": 0.4827},
    ("MLP", 1): {"accuracy": 0.8278, "f1": 0.8129, "precision": 0.8745},
    ("LSTM", 1): {"accuracy": 0.7224, "f1": 0.6677, "precision": 0.7450},
    ("Ours", 1): {"accuracy": 0.8740, "f1": 0.8650, "precision": 0.9233},
    ("FL", 2): {"accuracy": 0.8138, "f1": 0.8192, "precision": 0.8734},
    ("FD", 2): {"accuracy": 0.2012, "f1": 0.0853, "precision": 0.1037},
    ("DS-FL", 2): {"accuracy": 0.5353, "f1": 0.4395, "precision": 0.5745},
    ("MLP", 2): {"accuracy": 0.8285, "f1": 0.8131, "precision": 0.8757},
    ("LSTM", 2): {"accuracy": 0.6969, "f1": 0.6520, "precision": 0.6437},
    ("Ours", 2): {"accuracy": 0.8670, "f1": 0.8495, "precision": 0.9173},
    ("FL", 3): {"accuracy": 0.8113, "f1": 0.8164, "precision": 0.8670},
    ("FD", 3): {"accuracy": 0.5306, "f1": 0.4327, "precision": 0.4431},
    ("DS-FL", 3): {"accuracy": 0.2001, "f1": 0.0731, "precision": 0.1096},
    ("MLP", 3): {"accuracy": 0.8128, "f1": 0.7961, "precision": 0.8714},
    ("LSTM", 3): {"accuracy": 0.6080, "f1": 0.5808, "precision": 0.6086},
    ("Ours", 3): {"accuracy": 0.8422, "f1": 0.8247, "precision": 0.9017},
}

# --- Paper reference values (Table III, top-1 test accuracy % @ rounds 10/50/100/150/200) --------
TABLE_III_ROUNDS = [10, 50, 100, 150, 200]
PAPER_TABLE_III = {
    ("FL", 1): [39.31, 59.71, 68.00, 73.24, 73.79],
    ("FD", 1): [44.88, 47.21, 48.45, 48.54, 48.54],
    ("DS-FL", 1): [50.49, 50.49, 50.49, 50.49, 50.49],
    ("MLP", 1): [68.86, 78.28, 81.02, 81.87, 82.78],
    ("LSTM", 1): [37.31, 62.47, 68.77, 70.44, 72.24],
    ("Ours", 1): [77.90, 83.81, 84.90, 87.19, 87.40],
    ("FL", 2): [10.11, 28.66, 38.81, 48.00, 57.96],
    ("FD", 2): [20.10, 20.11, 20.11, 20.12, 20.12],
    ("DS-FL", 2): [30.76, 53.53, 53.53, 53.53, 53.53],
    ("MLP", 2): [26.84, 50.81, 81.53, 82.11, 82.85],
    ("LSTM", 2): [10.11, 41.29, 48.21, 55.76, 60.80],
    ("Ours", 2): [75.26, 80.43, 85.09, 86.31, 86.70],
    ("FL", 3): [10.22, 33.89, 45.41, 56.27, 63.35],
    ("FD", 3): [44.07, 47.56, 52.21, 53.06, 53.06],
    ("DS-FL", 3): [19.85, 19.93, 20.01, 20.01, 20.01],
    ("MLP", 3): [70.97, 76.74, 79.19, 80.13, 81.28],
    ("LSTM", 3): [10.11, 41.29, 48.21, 55.76, 60.80],
    ("Ours", 3): [72.16, 78.39, 83.28, 83.84, 84.22],
}

# --- Paper reference values (Table IV, communication cost MB @ accuracy) -------------------------
PAPER_TABLE_IV = {
    "FL": {
        "c_d0": None,
        1: {"c50": 15.81, "c75": 216.29, "c_top_acc": 1711.43, "top_acc": 0.8611},
        2: {"c50": 137.04, "c75": 514.14, "c_top_acc": 1745.06, "top_acc": 0.8138},
        3: {"c50": 110.69, "c75": 473.75, "c_top_acc": 1700.12, "top_acc": 0.8113},
    },
    "FD": {
        "c_d0": None,
        1: {"c50": None, "c75": None, "c_top_acc": 0.13, "top_acc": 0.4854},
        2: {"c50": None, "c75": None, "c_top_acc": 0.02, "top_acc": 0.2012},
        3: {"c50": None, "c75": None, "c_top_acc": 0.19, "top_acc": 0.5306},
    },
    "DS-FL": {
        "c_d0": 0.96,
        1: {"c50": 5.04, "c75": None, "c_top_acc": 5.04, "top_acc": 0.5049},
        2: {"c50": None, "c75": None, "c_top_acc": 22.63, "top_acc": 0.5353},
        3: {"c50": None, "c75": None, "c_top_acc": 46.35, "top_acc": 0.2001},
    },
    "Ours": {
        "c_d0": 0.96,
        1: {"c50": 0.01, "c75": 0.02, "c_top_acc": 0.55, "top_acc": 0.8740},
        2: {"c50": 0.01, "c75": 0.02, "c_top_acc": 0.49, "top_acc": 0.8670},
        3: {"c50": 0.01, "c75": 0.04, "c_top_acc": 0.47, "top_acc": 0.8422},
    },
}


@dataclass
class RunRecord:
    run_dir: Path
    run_id: str
    profile: str
    algorithm: str
    backbone: str
    scenario: int
    config: dict
    summary: dict
    metrics: pd.DataFrame
    comms: pd.DataFrame


def load_runs(runs_dir: Path) -> list[RunRecord]:
    records = []
    if not runs_dir.exists():
        return records
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        summary_path = d / "summary.json"
        config_path = d / "resolved_config.yaml"
        if not (summary_path.exists() and config_path.exists()):
            continue
        config = yaml.safe_load(config_path.read_text())
        summary = json.loads(summary_path.read_text())
        metrics_path = d / "metrics.parquet"
        comms_path = d / "communication.parquet"
        records.append(
            RunRecord(
                run_dir=d,
                run_id=d.name,
                profile=config["profile"],
                algorithm=config["algorithm"],
                backbone=config["backbone"],
                scenario=int(config["scenario"]),
                config=config,
                summary=summary,
                metrics=pd.read_parquet(metrics_path) if metrics_path.exists() else pd.DataFrame(),
                comms=pd.read_parquet(comms_path) if comms_path.exists() else pd.DataFrame(),
            )
        )
    return records


def _study(profile: str) -> str:
    for tag in ("ablation", "threshold", "label"):
        if tag in profile:
            return tag
    return "main"


def main_matrix_runs(records: list[RunRecord]) -> dict[tuple[str, int], RunRecord]:
    """One run per (method_label, scenario) cell -- prefers the run with more completed rounds, so
    a paper-scale run automatically supersedes an earlier smoke-scale stand-in for the same cell."""
    best: dict[tuple[str, int], RunRecord] = {}
    for r in records:
        if _study(r.profile) != "main":
            continue
        label = METHOD_LABELS.get((r.algorithm, r.backbone))
        if label is None:
            continue
        key = (label, r.scenario)
        cur = best.get(key)
        if cur is None or (r.summary.get("final_round") or 0) > (
            cur.summary.get("final_round") or 0
        ):
            best[key] = r
    return best


def _cumulative_comm_mb(comms: pd.DataFrame) -> pd.Series:
    """Cumulative real logical MB, summing all clients, directions, and phases."""
    per_round = comms.groupby("round")["logical_bytes"].sum().sort_index()
    return per_round.cumsum() / (1024 * 1024)


def _cumulative_paper_comm_mb(comms: pd.DataFrame) -> pd.Series:
    """Paper Table IV convention: representative-client uplink per communication round.

    Zhao et al.'s 0.01 MB SSFL hard-label value equals one open-set hard-label vector, not the
    federation-wide sum and not the downlink. ``paper_bytes`` also applies the paper's stated
    double precision to soft labels independently of the real Flower transport dtype.
    """
    uplink = comms[(comms["direction"] == "client_to_server") & (comms["phase"] == "train")]
    byte_column = "paper_bytes" if "paper_bytes" in uplink.columns else "logical_bytes"
    per_round = uplink.groupby("round")[byte_column].mean().sort_index()
    return per_round.cumsum() / (1024 * 1024)


def _comm_at_accuracy(metrics: pd.DataFrame, cum_mb: pd.Series, threshold: float) -> float | None:
    hits = metrics[metrics["accuracy"] >= threshold].sort_values("round")
    if hits.empty:
        return None
    return float(cum_mb.get(int(hits.iloc[0]["round"])))


def build_table_ii(records: list[RunRecord]) -> pd.DataFrame:
    cells = main_matrix_runs(records)
    rows = []
    for (label, scenario), ref in PAPER_TABLE_II.items():
        run = cells.get((label, scenario))
        repro = {"accuracy": None, "f1": None, "precision": None}
        if run is not None and not run.metrics.empty:
            last = run.metrics.sort_values("round").iloc[-1]
            repro = {
                "accuracy": float(last["accuracy"]),
                "f1": float(last["macro_f1"]),
                "precision": float(last["macro_precision"]),
            }
        rows.append(
            {
                "method": label,
                "scenario": scenario,
                "paper_accuracy": ref["accuracy"],
                "repro_accuracy": repro["accuracy"],
                "paper_f1": ref["f1"],
                "repro_f1": repro["f1"],
                "paper_precision": ref["precision"],
                "repro_precision": repro["precision"],
                "rounds_completed": int(run.summary["final_round"])
                if run and run.summary.get("final_round")
                else None,
                "run_id": run.run_id if run else None,
            }
        )
    return pd.DataFrame(rows).sort_values(["scenario", "method"]).reset_index(drop=True)


def build_table_iii(records: list[RunRecord]) -> pd.DataFrame:
    cells = main_matrix_runs(records)
    rows = []
    for (label, scenario), paper_vals in PAPER_TABLE_III.items():
        run = cells.get((label, scenario))
        row = {"method": label, "scenario": scenario}
        for round_n, paper_v in zip(TABLE_III_ROUNDS, paper_vals):
            repro_v = None
            if run is not None and not run.metrics.empty:
                match = run.metrics[run.metrics["round"] == round_n]
                if not match.empty:
                    repro_v = float(match.iloc[0]["accuracy"]) * 100
            row[f"paper_r{round_n}"] = paper_v
            row[f"repro_r{round_n}"] = repro_v
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["scenario", "method"]).reset_index(drop=True)


def build_table_iv(records: list[RunRecord]) -> pd.DataFrame:
    cells = main_matrix_runs(records)
    rows = []
    for label in ("FL", "FD", "DS-FL", "Ours"):
        for scenario in (1, 2, 3):
            ref = PAPER_TABLE_IV[label][scenario]
            run = cells.get((label, scenario))
            repro = {"c50": None, "c75": None, "c_top_acc": None, "top_acc": None}
            if run is not None and not run.metrics.empty and not run.comms.empty:
                cum_mb = _cumulative_paper_comm_mb(run.comms)
                repro["c50"] = _comm_at_accuracy(run.metrics, cum_mb, 0.50)
                repro["c75"] = _comm_at_accuracy(run.metrics, cum_mb, 0.75)
                best_row = run.metrics.loc[run.metrics["accuracy"].idxmax()]
                repro["top_acc"] = float(best_row["accuracy"])
                repro["c_top_acc"] = float(cum_mb.get(int(best_row["round"])))
            rows.append(
                {
                    "method": label,
                    "scenario": scenario,
                    "paper_c_d0_mb": PAPER_TABLE_IV[label]["c_d0"],
                    "paper_c50_mb": ref["c50"],
                    "repro_c50_mb": repro["c50"],
                    "paper_c75_mb": ref["c75"],
                    "repro_c75_mb": repro["c75"],
                    "paper_c_top_acc_mb": ref["c_top_acc"],
                    "repro_c_top_acc_mb": repro["c_top_acc"],
                    "paper_top_acc": ref["top_acc"],
                    "repro_top_acc": repro["top_acc"],
                    "run_id": run.run_id if run else None,
                }
            )
    return pd.DataFrame(rows).sort_values(["scenario", "method"]).reset_index(drop=True)


def _ablation_label(config: dict) -> str:
    disc, vote = config["ssfl_discriminator_mode"], config["ssfl_voting_mode"]
    if disc == "simple_filter":
        return "Simply Filtering"
    if disc == "enabled" and vote == "enabled":
        return "Ours"
    if disc == "disabled" and vote == "enabled":
        return "Ours w/o Discriminating"
    if disc == "enabled" and vote == "disabled":
        return "Ours w/o Voting"
    return "Ours w/o Discriminating and Voting"


def _threshold_label(config: dict) -> str:
    policy = config["ssfl_threshold_policy"]
    if policy == "median":
        return "Confidence Threshold Median"
    return f"Confidence Threshold {policy.replace('fixed_', '').replace('_', '.')}"


def _label_study_label(config: dict) -> str:
    if config["ssfl_label_representation"] == "hard":
        return "Hard Label"
    return f"Soft Label w. {config['ssfl_soft_label_round_decimals']}f"


def _study_records(records: list[RunRecord], tag: str) -> list[RunRecord]:
    return [r for r in records if _study(r.profile) == tag]


def fig_allocation(data_dir: Path, output_dir: Path) -> Path | None:
    scenario_files = (
        sorted((data_dir / "scenarios").glob("*_allocation_stats.json"))
        if (data_dir / "scenarios").exists()
        else []
    )
    if not scenario_files:
        return None
    fig, axes = plt.subplots(
        1, len(scenario_files), figsize=(6 * len(scenario_files), 5), squeeze=False
    )
    for ax, path in zip(axes[0], scenario_files):
        stats = json.loads(path.read_text())
        for ci, client in enumerate(stats["clients"], start=1):
            for class_id, count in enumerate(client["class_counts"]):
                if count:
                    ax.scatter(ci, class_id, s=max(count / 5, 4), color="tab:blue", alpha=0.7)
        ax.set_xlabel("Client ID")
        ax.set_ylabel("Class Label")
        ax.set_title(path.stem.replace("_allocation_stats", ""))
        ax.set_yticks(range(11))
    fig.suptitle("Fig 2: samples per class allocated to each client")
    fig.tight_layout()
    out = output_dir / "fig2_allocation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_confusion_matrices(records: list[RunRecord], output_dir: Path) -> Path | None:
    cells = main_matrix_runs(records)
    scenarios = [1, 2, 3]
    runs = [cells.get(("Ours", s)) for s in scenarios]
    if not any(r is not None and (r.run_dir / "confusion_matrices.npz").exists() for r in runs):
        return None
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, scenario, run in zip(axes, scenarios, runs):
        ax.set_title(f"Scenario {scenario}")
        if run is None or not (run.run_dir / "confusion_matrices.npz").exists():
            ax.axis("off")
            continue
        npz = np.load(run.run_dir / "confusion_matrices.npz")
        last_round = max(int(k.split("_")[1]) for k in npz.files)
        cm = npz[f"round_{last_round}"].astype(float)
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(11))
        ax.set_xticklabels(CLASS_NAMES, rotation=90)
        ax.set_yticks(range(11))
        ax.set_yticklabels(CLASS_NAMES)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Fig 3: SSFL confusion matrices (test set, final round)")
    fig.tight_layout()
    out = output_dir / "fig3_confusion_matrices.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_accuracy_curves(
    records: list[RunRecord], tag: str, label_fn, title: str, out_name: str, output_dir: Path
) -> Path | None:
    runs = _study_records(records, tag)
    if not runs:
        return None
    by_scenario: dict[int, list[RunRecord]] = {}
    for r in runs:
        by_scenario.setdefault(r.scenario, []).append(r)
    scenarios = sorted(by_scenario)
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 5), squeeze=False)
    for ax, scenario in zip(axes[0], scenarios):
        for r in sorted(by_scenario[scenario], key=lambda r: label_fn(r.config)):
            if r.metrics.empty:
                continue
            m = r.metrics.sort_values("round")
            ax.plot(m["round"], m["accuracy"] * 100, marker="o", label=label_fn(r.config))
        ax.set_title(f"Scenario {scenario}")
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Test Accuracy (%)")
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    out = output_dir / out_name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_label_study(records: list[RunRecord], output_dir: Path) -> Path | None:
    runs = _study_records(records, "label")
    if not runs:
        return None
    by_scenario: dict[int, list[RunRecord]] = {}
    for r in runs:
        by_scenario.setdefault(r.scenario, []).append(r)
    scenarios = sorted(by_scenario)
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(12, 5 * len(scenarios)), squeeze=False)
    for row, scenario in enumerate(scenarios):
        ax_acc, ax_comm = axes[row]
        for r in sorted(by_scenario[scenario], key=lambda r: _label_study_label(r.config)):
            if r.metrics.empty:
                continue
            m = r.metrics.sort_values("round")
            label = _label_study_label(r.config)
            ax_acc.plot(m["round"], m["accuracy"] * 100, marker="o", label=label)
            if not r.comms.empty:
                cum_mb = _cumulative_paper_comm_mb(r.comms)
                rounds = sorted(cum_mb.index)
                ax_comm.plot(rounds, [cum_mb[x] for x in rounds], marker="o", label=label)
        ax_acc.set_title(f"Scenario {scenario}: Test Accuracy")
        ax_acc.set_xlabel("Communication Round")
        ax_acc.set_ylabel("Test Accuracy (%)")
        ax_acc.legend(fontsize=8)
        ax_comm.set_title(f"Scenario {scenario}: Communication Cost")
        ax_comm.set_xlabel("Communication Round")
        ax_comm.set_ylabel("Communication Cost (MB)")
        ax_comm.legend(fontsize=8)
    fig.suptitle("Fig 6: label representation study")
    fig.tight_layout()
    out = output_dir / "fig6_label_study.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _to_markdown_table(df: pd.DataFrame) -> str:
    """Tiny hand-rolled Markdown table writer -- ``pandas.DataFrame.to_markdown`` needs the
    ``tabulate`` package, which isn't in this project's dependencies and isn't worth adding for
    this alone."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append("")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_markdown_report(
    output_dir: Path,
    records: list[RunRecord],
    table_ii: pd.DataFrame,
    table_iii: pd.DataFrame,
    table_iv: pd.DataFrame,
    figures: list[Path],
) -> None:
    lines = ["# SSFL reproduction report", ""]
    lines.append(
        f"Generated from {len(records)} run director{'y' if len(records) == 1 else 'ies'} under `artifacts/runs/`."
    )
    lines.append("")
    lines.append("## Table II -- Accuracy / F1 / Precision (reproduced vs. paper)")
    lines.append("")
    lines.append(_to_markdown_table(table_ii))
    lines.append("")
    lines.append("## Table III -- Top-1 test accuracy (%) @ communication round")
    lines.append("")
    lines.append(_to_markdown_table(table_iii))
    lines.append("")
    lines.append("## Table IV -- Communication cost (MB) @ test accuracy")
    lines.append("")
    lines.append(_to_markdown_table(table_iv))
    lines.append("")
    if figures:
        lines.append("## Figures")
        lines.append("")
        for path in figures:
            lines.append(f"![{path.stem}]({path.name})")
        lines.append("")
    lines.append("## Deviations and assumptions")
    lines.append("")
    lines.append(
        "- `repro_*` / `Fig` cells are blank where no run for that `(method, scenario)` or study "
        "variant has been executed yet -- paper-scale 200-round runs are gated on GPU hardware "
        "(REPRODUCIBILITY.md); only CPU smoke-scale runs may be present."
    )
    lines.append(
        "- Table IV's `paper_c_d0_mb` (cost of distributing the open dataset) has no reproduced "
        "counterpart: this simulation reads the open set from shared local storage per client "
        "rather than transmitting it as a Flower `Message`, so it never appears in "
        "`communication.parquet`."
    )
    lines.append(
        "- Table IV follows the paper's representative-client train-uplink convention using "
        "`paper_bytes` (including its stated double-precision soft labels). The complete real "
        "logical and serialized traffic for every client/direction/phase remains available in "
        "`communication.parquet`."
    )
    (output_dir / "report.md").write_text("\n".join(lines))


def build_report(runs_dir: Path, output_dir: Path, data_dir: Path = Path("artifacts/data")) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_runs(runs_dir)

    table_ii = build_table_ii(records)
    table_iii = build_table_iii(records)
    table_iv = build_table_iv(records)
    for name, df in [("table_ii", table_ii), ("table_iii", table_iii), ("table_iv", table_iv)]:
        df.to_csv(output_dir / f"{name}.csv", index=False)
        df.to_parquet(output_dir / f"{name}.parquet", index=False)

    figures = [
        p
        for p in [
            fig_allocation(data_dir, output_dir),
            fig_confusion_matrices(records, output_dir),
            fig_accuracy_curves(
                records,
                "ablation",
                _ablation_label,
                "Fig 4: ablation study",
                "fig4_ablation.png",
                output_dir,
            ),
            fig_accuracy_curves(
                records,
                "threshold",
                _threshold_label,
                "Fig 5: confidence threshold study",
                "fig5_threshold.png",
                output_dir,
            ),
            fig_label_study(records, output_dir),
        ]
        if p is not None
    ]

    _write_markdown_report(output_dir, records, table_ii, table_iii, table_iv, figures)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the SSFL Tables II-IV / Figures 2-6 report from artifacts/runs."
    )
    parser.add_argument("--runs", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/report"))
    args = parser.parse_args()
    build_report(args.runs, args.output, args.data_dir)
    print(f"[build_report] wrote report to {args.output}")


if __name__ == "__main__":
    main()
