# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Reproduction of *"Semisupervised Federated-Learning-Based Intrusion Detection Method for Internet
of Things"* on the N-BaIoT dataset: four federated protocols (SSFL, FL, FD, DS-FL) compared across
three non-IID client-partition scenarios, built on Flower's message-based API + PyTorch.

- `SSFL_IMPLEMENTATION_PLAN.md` — the spec this repo implements (architecture/protocol detail is
  still accurate background reading; its milestone numbering is stale, see below).
- `REPRODUCIBILITY.md` — every assumption, deviation, and gotcha vs. the paper, numbered (#1-#30).
  **Read this before touching config resolution, `Strategy.start()` overrides, seeding, or
  anything `flwr run`-related** — most non-obvious behavior in this codebase is explained there,
  including several bugs that were found only by live execution, not code review.
- `DATA_CARD.md` / `MODEL_CARD.md` / `SECURITY.md` — dataset provenance, model architectures, and
  the threat model / verified privacy boundaries per protocol.
- Work is tracked via the task tool's TaskList (M0-M12 milestones), not by any file in the repo.

## Commands

```bash
uv sync                                                      # install deps (uv.lock is authoritative)

# one-time data prep (raw N-BaIoT CSVs -> artifacts/data/, gitignored)
uv run python -m ssfl.data.prepare_data --input data --output artifacts/data --seed 2023

uv run pytest -q                                             # unit + protocol/security tests (~45s)
uv run pytest -m slow -q                                     # + real flwr simulations & subprocess tests (minutes)
uv run pytest tests/unit/test_models.py -q                   # single file
uv run pytest tests/unit/test_models.py::test_cnn_shape -q   # single test
uv run ruff check .                                          # lint

# one federated run (algorithm: ssfl|fl|fd|dsfl; scenario: 1|2|3; profile: smoke|paper|...)
uv run flwr run . --run-config 'profile="smoke" algorithm="ssfl" scenario=1 device="cpu"'

uv run python -m ssfl.experiments.run_suite --matrix configs/experiments_smoke.yaml   # multi-run matrix
uv run python -m ssfl.reporting.build_report --runs artifacts/runs --output artifacts/report
```

`profile="paper"` runs the real 200-round config — it's fully written but gated on GPU hardware and
not executed in this build environment; the CPU `smoke` profile (tiny samples, 2 rounds, few
clients) is the actual verification gate here. Never report a `paper`-profile run as executed
unless it really was run on real hardware.

Real (non-simulation) SuperLink/SuperNode deployment lives under `deployment/` — see
`deployment/README.md`, separate from the `flwr run` Simulation/Ray path used everywhere else.

### Environment gotchas

- **The absolute repo path must not contain a space.** This checkout is literally
  `.../Federated learning with psuedo labeling/...`; Flower's local-simulation SuperLink builds an
  Alembic config by joining paths with a plain space, which Alembic then re-splits on whitespace —
  a spaced path silently discovers zero migration scripts and every `flwr run` fails opaquely with
  `no such table: fab`. Workaround already applied: `.venv` is a symlink to a space-free path
  (`~/.venvs/ssfl`). If `.venv` is ever recreated from scratch (`uv venv` / fresh `uv sync`),
  re-apply the move+symlink — see `REPRODUCIBILITY.md`'s host-environment section for the exact
  mechanism.
- No hard dependency pins — `uv sync` resolves "latest installable" against `requires-python =
  ">=3.12"` by deliberate choice (not the paper spec's literal `torch==2.13.0`/`flwr==1.32.1`).
  Don't "fix" `pyproject.toml` back to pinned versions.
- CPU thread count is forced to 1 per process (`device.py::_cpu_device` calls
  `torch.set_num_threads(1)`) — many concurrent simulated clients each spawning a full OMP pool
  oversubscribes the machine ~N× for this model's tiny tensors. Don't remove this without rereading
  REPRODUCIBILITY.md #17.
- Before starting any Docker/colima build in this repo, check `df -h` first — a `docker compose
  build` here has previously filled the host disk to 99% (REPRODUCIBILITY.md #30).

## Architecture

### Two independent entrypoints, two config models (`src/ssfl/config.py`)

- `DataPrepConfig` — standalone, consumed only by `ssfl.data.prepare_data` (argparse CLI). Produces
  `artifacts/data/` (private/open/test splits, scaler, per-scenario client-assignment manifests,
  `dataset_manifest.json` with a content hash).
- `ExperimentConfig` — consumed by `flwr run` via `--run-config`, by `run_suite.py`, and by the
  deployment CLIs. Deliberately **flat** (Flower's `run-config` is a flat `key=value` string);
  grouping is by name prefix (`ssfl_*`, `dsfl_*`) rather than nesting. Both models set
  `extra="forbid"` so unknown keys fail fast, before any Flower/training code runs.
- **`configs/<profile>.yaml` is the unit of config** (`smoke`, `paper`, `deployment`, ablation
  variants). `pyproject.toml`'s `[tool.flwr.app.config]` intentionally declares only `profile` +
  a tiny allowlist (`algorithm`/`scenario`/`device`) meant to be picked per-invocation — Flower
  populates `run_config` from *every* key in that table on *every* run, so adding any
  profile-defining field there (e.g. `num-server-rounds`) would silently clobber every non-`paper`
  profile YAML. **Express a new configuration as a new `configs/<name>.yaml`, never as a new key in
  `pyproject.toml`.** (Full rationale: REPRODUCIBILITY.md #18.)
- `experiment_config_from_run_config()` is the actual runtime loader both `client_app.py` and
  `server_app.py` call: layers `configs/<profile>.yaml` (or `artifacts/generated_configs/<profile>.yaml`,
  where `run_suite.py` writes one resolved profile per matrix entry for knobs Flower's CLI
  allowlist can't carry) underneath the flat `run_config` overrides.

### Protocol implementation split: `protocols/` (pure) vs `strategies/` (Flower-facing)

- `protocols/{ssfl,fl,fd,dsfl}.py` — the actual per-algorithm training/aggregation math (client
  proposal/distillation steps, server aggregation, loss composition). No `flwr` import in the parts
  that can avoid it, so aggregation logic is unit-testable without starting Flower.
- `protocols/message.py` — Flower-independent `Envelope`/`ExpectedContext`/`validate_envelope`:
  the safeguards shared by all four protocols (protocol-version check, algorithm/scenario/round/
  phase match, sender allowlist, message-id dedup). Algorithm-specific payload shape/dtype checks
  live next to the aggregation code that consumes them instead.
- `strategies/{ssfl,fd,dsfl}.py` — custom Flower `Strategy` subclasses. SSFL/FD/DS-FL never
  transmit model parameters (they exchange labels/logits/probabilities instead), so they override
  `Strategy.start()` directly to run **two** `Grid.send_and_receive()` exchanges per paper
  communication round (proposal + distillation), rather than reusing the standard single
  configure_train/aggregate_train shape. FL is the exception: it reuses Flower's built-in `FedAvg`
  unmodified since it's a stock parameter-averaging pattern.
- `client_app.py` / `server_app.py` are the single `ClientApp`/`ServerApp` Flower entrypoints
  (declared in `pyproject.toml`'s `[tool.flwr.app.components]`); both dispatch on
  `exp_config.algorithm` to the right protocol/strategy. SSFL/FD/DS-FL persist client model weights
  across rounds in `context.state` (never on the wire); FL's model arrives fresh every round instead.
  Round-0 weight init never crosses the wire either — every client independently derives the same
  initial weights via `seed_everything(seed [+ offset])` from the shared config seed.

### Data pipeline (`data/`)

`prepare_data.py` orchestrates: `discovery` (find the 89 flat N-BaIoT CSVs) → `sampling` (mini
dataset per device/class) → `partition` (per-scenario non-IID client shard/Dirichlet allocation,
McMahan-style — see REPRODUCIBILITY.md #13 for the exact algorithm per scenario) → `scaling`
(min-max fit per `NormalizationMode`) → `labels`/`manifest`/`io` (label map, content-hashed
manifest, on-disk layout) → `archive`. `datasets.py` is what `client_app.py`/`server_app.py`
actually load at runtime (`load_client_private_data`, `load_open_data`, `load_test_data`,
`load_client_assignments`).

### Run lifecycle & reproducibility bundle

`run_context.py::RunContext.create()` writes `resolved_config.yaml` / `environment.json` /
`dataset_manifest.json` / `code_version.json` into `artifacts/runs/<run_id>/` **before** training
starts. `<run_id>` is a deterministic hash over config + dataset manifest + git commit + seed
(`config.py::compute_run_id`). `metrics.parquet` (`metrics.py`) and `communication.parquet`
(`comms.py::CommsTrackingStrategy`, which wraps whichever `Strategy` is active) are written after
`strategy.start()` returns. `events.jsonl` is a structured log stream (`logging_utils.py`) bound
with run identity and written per-round. Note: mid-run checkpoint/resume is scaffolded
(`RunContext.resume`, `checkpoint_rounds` config) but not actually wired into `server_app.py` yet —
an interrupted run restarts from round 1, not from its last checkpoint (REPRODUCIBILITY.md #27).

`experiments/run_suite.py` drives a matrix of `configs/experiments*.yaml` entries as separate
`flwr run` subprocesses (`--stream` is required — without it the CLI detaches immediately instead
of blocking); `reporting/build_report.py` aggregates `artifacts/runs/*/summary.json` +
`metrics.parquet`/`communication.parquet` into the paper's comparison tables/figures.

### Testing layout

- `tests/unit/` — fast, no Flower/subprocess (data prep, models, config, logging, run_suite/build_report logic).
- `tests/protocol/` — per-algorithm aggregation + message-contract correctness (`test_ssfl.py`,
  `test_fd.py`, etc.), independent of Flower wiring.
- `tests/privacy/` — wire-level payload checks (nothing leaks beyond the documented protocol contract).
- `tests/integration/`, `tests/deployment/` — real `flwr run` simulations / real SuperLink+SuperNode
  subprocesses; marked `@pytest.mark.slow` and excluded by default (`addopts = "-m 'not slow'"` in
  `pyproject.toml`). Run explicitly with `pytest -m slow`.
- `tests/conftest.py`'s `prepared_data_root` fixture builds a synthetic N-BaIoT-shaped CSV tree and
  runs the real `prepare_data.run_full()` against it — shared by privacy/integration/deployment
  tests that need a real prepared-data tree rather than fabricated arrays.
