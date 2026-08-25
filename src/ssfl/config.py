"""Typed configuration for the SSFL reproduction: data-prep config and per-run experiment config.

Two independent config surfaces, matching the two independent entrypoints:

- ``DataPrepConfig`` — consumed only by ``ssfl.data.prepare_data`` (argparse CLI), never by Flower.
- ``ExperimentConfig`` — consumed by ``flwr run`` (via ``--run-config``), ``run_suite``, and the
  deployment CLIs. Deliberately flat (not deeply nested) because Flower's ``run-config`` mechanism
  is a flat ``key=value`` string; nesting would require a lossy flatten/unflatten translation layer
  for every field. Grouping is expressed through name prefixes (``ssfl_*``) instead of nesting.

Both models set ``extra="forbid"`` so unknown keys fail fast, and both are validated before any
Flower or training code runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Algorithm(str, Enum):
    ssfl = "ssfl"
    fl = "fl"
    fd = "fd"
    dsfl = "dsfl"


class Backbone(str, Enum):
    cnn = "cnn"
    mlp = "mlp"
    lstm = "lstm"


class Scenario(int, Enum):
    one = 1
    two = 2
    three = 3


class NormalizationMode(str, Enum):
    all_mini = "all_mini"
    private_only = "private_only"


class ThresholdPolicy(str, Enum):
    median = "median"
    fixed_0_7 = "fixed_0_7"
    fixed_0_8 = "fixed_0_8"
    fixed_0_9 = "fixed_0_9"

    @property
    def fixed_value(self) -> float | None:
        return {
            ThresholdPolicy.fixed_0_7: 0.7,
            ThresholdPolicy.fixed_0_8: 0.8,
            ThresholdPolicy.fixed_0_9: 0.9,
        }.get(self)


class DiscriminatorMode(str, Enum):
    enabled = "enabled"
    disabled = "disabled"
    simple_filter = "simple_filter"


class VotingMode(str, Enum):
    enabled = "enabled"
    disabled = "disabled"


class LabelRepresentation(str, Enum):
    hard = "hard"
    soft = "soft"


class DeviceKind(str, Enum):
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"
    auto = "auto"


class RunKind(str, Enum):
    """Distinguishes canonical paper reproduction from research-extension runs.

    Kept as its own field (not inferred) so paper tables can filter on it directly and an
    extension run can never silently masquerade as a paper-comparable result.
    """

    paper = "paper"
    extension = "extension"


# ---------------------------------------------------------------------------
# Data preparation config (standalone; no Flower/training coupling)
# ---------------------------------------------------------------------------


class DataPrepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_path: Path
    output_path: Path = Path("artifacts/data")
    seed: int = 2023
    samples_per_subset: int = 1000
    private_ratio: float = 0.7
    open_ratio: float = 0.1
    test_ratio: float = 0.2
    normalization_mode: NormalizationMode = NormalizationMode.all_mini
    dirichlet_alpha: float = 0.1
    validate_only: bool = False

    @model_validator(mode="after")
    def _check_ratios(self) -> "DataPrepConfig":
        total = self.private_ratio + self.open_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")
        private_n = round(self.samples_per_subset * self.private_ratio)
        open_n = round(self.samples_per_subset * self.open_ratio)
        test_n = round(self.samples_per_subset * self.test_ratio)
        if private_n + open_n + test_n != self.samples_per_subset:
            raise ValueError(
                "split ratios do not exactly partition samples_per_subset "
                f"({private_n}+{open_n}+{test_n} != {self.samples_per_subset})"
            )
        return self

    @property
    def split_counts(self) -> tuple[int, int, int]:
        """(private, open, test) row counts per device/class subset."""
        return (
            round(self.samples_per_subset * self.private_ratio),
            round(self.samples_per_subset * self.open_ratio),
            round(self.samples_per_subset * self.test_ratio),
        )


# ---------------------------------------------------------------------------
# Experiment config (flat; consumed via flwr run-config)
# ---------------------------------------------------------------------------

# Valid (label_representation, voting_mode) pairs, matching the five documented SSFL ablations
# (full / no-discriminator / simple-filtering use hard+voting; no-voting / no-discriminator-and-
# voting use soft+masked-average). Anything else is not a paper- or spec-defined configuration.
_VALID_LABEL_VOTING_PAIRS = {
    (LabelRepresentation.hard, VotingMode.enabled),
    (LabelRepresentation.soft, VotingMode.disabled),
}


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # --- identity -----------------------------------------------------
    profile: str = "paper"
    run_kind: RunKind = RunKind.paper
    algorithm: Algorithm = Algorithm.ssfl
    backbone: Backbone = Backbone.cnn
    scenario: Scenario = Scenario.one
    seed: int = 2023

    # --- data ------------------------------------------------------------
    data_path: Path = Path("artifacts/data")
    output_path: Path = Path("artifacts/runs")

    # --- training --------------------------------------------------------
    num_server_rounds: int = Field(default=200, alias="num-server-rounds")
    local_epochs: int = 5
    batch_size: int = 80
    paper_text_batch: bool = (
        False  # True => use batch size 100 (Section V-C compatibility override)
    )
    learning_rate: float = 1e-4
    device: DeviceKind = DeviceKind.auto
    deterministic: bool = True

    # --- SSFL-specific -----------------------------------------------------
    ssfl_threshold_policy: ThresholdPolicy = ThresholdPolicy.median
    ssfl_discriminator_mode: DiscriminatorMode = DiscriminatorMode.enabled
    ssfl_voting_mode: VotingMode = VotingMode.enabled
    ssfl_label_representation: LabelRepresentation = LabelRepresentation.hard
    ssfl_soft_label_round_decimals: int | None = None

    # --- DS-FL-specific ------------------------------------------------
    dsfl_temperature: float = 0.1

    # --- checkpointing / logging ----------------------------------------
    checkpoint_interval: int = 10
    checkpoint_rounds: tuple[int, ...] = (10, 50, 100, 150, 200)
    logging_interval: int = 1
    log_every_batch: bool = False
    system_monitor_interval_seconds: float = 1.0
    resume_from: Path | None = None

    # --- resources (simulation) -----------------------------------------
    client_num_cpus: int = 1
    client_num_gpus: float = 0.0
    max_concurrent_clients: int = 8

    @property
    def effective_batch_size(self) -> int:
        return 100 if self.paper_text_batch else self.batch_size

    @model_validator(mode="after")
    def _check_ssfl_combinations(self) -> "ExperimentConfig":
        if self.algorithm != Algorithm.ssfl:
            return self

        pair = (self.ssfl_label_representation, self.ssfl_voting_mode)
        if pair not in _VALID_LABEL_VOTING_PAIRS:
            raise ValueError(
                "invalid ssfl_label_representation/ssfl_voting_mode combination "
                f"{pair}; hard labels require voting=enabled, soft labels require voting=disabled "
                "(matches the paper's five documented ablations)"
            )
        if self.ssfl_label_representation == LabelRepresentation.soft:
            if self.ssfl_soft_label_round_decimals not in (2, 4, 6, 8):
                raise ValueError(
                    "soft label representation requires ssfl_soft_label_round_decimals in "
                    "{2, 4, 6, 8}"
                )
        else:
            if self.ssfl_soft_label_round_decimals is not None:
                raise ValueError(
                    "ssfl_soft_label_round_decimals must be unset when using hard labels"
                )
        return self

    @model_validator(mode="after")
    def _check_backbone_algorithm(self) -> "ExperimentConfig":
        if self.backbone != Backbone.cnn and self.algorithm != Algorithm.ssfl:
            raise ValueError(
                f"backbone={self.backbone.value} is only defined for algorithm=ssfl "
                "(FL/FD/DS-FL baselines use the paper CNN exclusively)"
            )
        return self

    @model_validator(mode="after")
    def _check_checkpoint_rounds(self) -> "ExperimentConfig":
        bad = [r for r in self.checkpoint_rounds if r > self.num_server_rounds]
        if bad and self.run_kind == RunKind.paper:
            raise ValueError(
                f"checkpoint_rounds {bad} exceed num_server_rounds={self.num_server_rounds}"
            )
        return self

    @model_validator(mode="after")
    def _check_cuda_resources(self) -> "ExperimentConfig":
        if self.device == DeviceKind.cuda and self.client_num_gpus <= 0:
            raise ValueError("device=cuda requires client_num_gpus > 0 so Ray exposes the GPU")
        if self.client_num_gpus < 0 or self.client_num_gpus > 1:
            raise ValueError("client_num_gpus must be in [0, 1]")
        if self.max_concurrent_clients < 1:
            raise ValueError("max_concurrent_clients must be >= 1")
        if self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be >= 1")
        if self.logging_interval < 1:
            raise ValueError("logging_interval must be >= 1")
        if self.system_monitor_interval_seconds <= 0:
            raise ValueError("system_monitor_interval_seconds must be > 0")
        return self

    def config_hash(self) -> str:
        """Stable hash over the resolved, user-controllable config (excludes filesystem paths that
        legitimately vary by machine: data_path/output_path/resume_from)."""
        payload = self.model_dump(mode="json", exclude={"data_path", "output_path", "resume_from"})
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def num_clients(self) -> int:
        return {Scenario.one: 27, Scenario.two: 89, Scenario.three: 89}[self.scenario]


# ---------------------------------------------------------------------------
# Loading / overrides
# ---------------------------------------------------------------------------


def _normalize_key(key: str) -> str:
    return key.replace("-", "_")


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with open(path, "r") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a top-level mapping, got {type(data).__name__}")
    return data


def parse_run_config_string(run_config: str) -> dict[str, str]:
    """Parse Flower's flat ``--run-config "k=v k2=v2"`` syntax into a string dict.

    Values are left as strings; pydantic performs the actual type coercion during model
    validation, which is also where a malformed value surfaces as a clear error.
    """
    overrides: dict[str, str] = {}
    for token in run_config.split():
        if "=" not in token:
            raise ValueError(f"malformed run-config token {token!r}, expected key=value")
        key, _, value = token.partition("=")
        overrides[_normalize_key(key)] = value
    return overrides


def load_experiment_config(
    profile_path: Path,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Load a named YAML profile from ``configs/`` and apply CLI/run-config overrides on top.

    Unknown keys anywhere in this chain raise ``pydantic.ValidationError`` before any Flower or
    training code runs, satisfying the "fail before Flower starts" requirement.
    """
    base = load_yaml(profile_path)
    base = {_normalize_key(k): v for k, v in base.items()}
    if overrides:
        base.update({_normalize_key(k): v for k, v in overrides.items()})
    return ExperimentConfig.model_validate(base)


def _resolve_runtime_paths(config: ExperimentConfig, repo_root: Path) -> ExperimentConfig:
    """Anchor profile paths to the app root before Flower changes the runtime working directory."""
    updates: dict[str, Path] = {}
    for field in ("data_path", "output_path", "resume_from"):
        path = getattr(config, field)
        if path is not None and not path.is_absolute():
            updates[field] = (repo_root / path).resolve()
    return config.model_copy(update=updates)


def experiment_config_from_run_config(run_config: dict[str, Any]) -> ExperimentConfig:
    """Build an :class:`ExperimentConfig` from a Flower ``Context.run_config`` dict (the
    hyphenated flat mapping ``ClientApp``/``ServerApp`` receive at runtime).

    Layers the ``configs/<profile>.yaml`` matching ``run_config``'s ``profile`` key (default
    ``paper``) underneath the flat overrides -- the same merge ``load_experiment_config`` does for
    the CLI path -- so profile-only fields like ``checkpoint_rounds`` stay consistent with
    ``num_server_rounds`` without needing to be repeated in every ``--run-config`` string. Falls
    back to ``artifacts/generated_configs/<profile>.yaml`` (where ``experiments/run_suite.py``
    writes one resolved profile per matrix entry, since Flower's ``--run-config`` allowlist can't
    carry the ablation/threshold/label-study knobs directly), then to validating the overrides
    alone if neither file exists (e.g. a custom profile name meant to be fully specified via
    overrides).
    """
    overrides = {_normalize_key(k): v for k, v in run_config.items()}
    # TOML has no null value, so pyproject exposes an empty-string invocation default. Treat it as
    # absent while preserving a real CLI path (`resume-from="artifacts/runs/<run-id>"`).
    if overrides.get("resume_from") == "":
        overrides.pop("resume_from")
    profile = overrides.get("profile", "paper")
    repo_root = Path(__file__).resolve().parent.parent.parent
    for candidate in (
        repo_root / "configs" / f"{profile}.yaml",
        repo_root / "artifacts" / "generated_configs" / f"{profile}.yaml",
    ):
        if candidate.exists():
            return _resolve_runtime_paths(load_experiment_config(candidate, overrides), repo_root)
    return _resolve_runtime_paths(ExperimentConfig.model_validate(overrides), repo_root)


def load_data_prep_config(overrides: dict[str, Any]) -> DataPrepConfig:
    return DataPrepConfig.model_validate(overrides)


# ---------------------------------------------------------------------------
# Environment snapshot
# ---------------------------------------------------------------------------


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
        return out.stdout.strip()
    except Exception:
        return None


def capture_environment_snapshot() -> dict[str, Any]:
    """Best-effort snapshot of interpreter, dependency versions, device, and code state.

    Written into every run directory (``environment.json``) alongside the resolved config, per the
    reproducibility-package requirement. Never raises: a missing optional dependency degrades to a
    ``null`` field rather than failing the run.
    """
    import torch

    def _version(module_name: str) -> str | None:
        try:
            mod = __import__(module_name)
            return getattr(mod, "__version__", None)
        except Exception:
            return None

    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": f"{properties.major}.{properties.minor}",
                    "multi_processor_count": properties.multi_processor_count,
                }
            )
    try:
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
        )
    except Exception:
        git_dirty = None
    try:
        nvidia_smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name,uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except Exception:
        nvidia_smi = None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        physical_pages = os.sysconf("SC_PHYS_PAGES")
        ram_bytes = int(page_size * physical_pages)
    except (AttributeError, ValueError, OSError):
        ram_bytes = None

    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_bytes": ram_bytes,
        "git_commit": _git_commit(),
        "git_dirty": git_dirty,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_devices": cuda_devices,
        "nvidia_smi": nvidia_smi,
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "mps_available": torch.backends.mps.is_available(),
        "dependency_versions": {
            name: _version(name)
            for name in ["flwr", "ray", "pandas", "pyarrow", "sklearn", "pydantic", "numpy"]
        },
    }


def compute_run_id(config: ExperimentConfig, dataset_manifest_hash: str | None = None) -> str:
    """Stable run ID derived from resolved config, dataset lineage, code version, and seed."""
    parts = {
        "config_hash": config.config_hash(),
        "dataset_manifest_hash": dataset_manifest_hash,
        "git_commit": _git_commit(),
        "seed": config.seed,
    }
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    return f"{config.algorithm.value}-s{config.scenario.value}-{config.profile}-{digest}"
