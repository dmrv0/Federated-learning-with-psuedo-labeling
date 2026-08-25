# SSFL Paper Reproduction — Expanded Implementation Plan

## 1. Target Outcome

Deliver four related capabilities from one codebase:

1. **Faithful paper reproduction**
   - SSFL, FL, FD, DS-FL.
   - Three non-IID scenarios.
   - CNN, MLP, and LSTM comparisons.
   - All ablations and sensitivity studies.
   - Tables II-IV and Figures 2-6.

2. **Research experimentation platform**
   - Deterministic experiment configurations.
   - Multiple seeds and uncertainty reporting.
   - Checkpoint/resume.
   - Reusable algorithm and partition interfaces.
   - Clear separation between paper-defined behavior and implementation assumptions.

3. **Deployable Flower application**
   - Identical learning code for simulation and network deployment.
   - Stateful clients.
   - Authenticated and encrypted communication.
   - Client lifecycle, failure, and timeout handling.
   - Container and Kubernetes deployment options.

4. **Auditable reproducibility package**
   - Dataset lineage and checksums.
   - Dependency lock.
   - Experiment manifests.
   - Privacy and communication audits.
   - Generated reports and archived run bundles.

Full 200-round GPU experiments remain gated on user-provided compute. CPU smoke verification is the implementation acceptance gate.

---

## 2. System Architecture

```text
Raw /data
    │
    ▼
Standalone preparation pipeline
    │
    ├── validated Mini-N-BaIoT
    ├── private/open/test splits
    ├── scenario manifests
    ├── scaler and schema
    └── lineage/checksum manifest
             │
             ▼
      Shared training core
      ├── models
      ├── optimizers/losses
      ├── metrics
      └── protocol primitives
             │
       ┌─────┴─────┐
       ▼           ▼
Flower ClientApp  Flower ServerApp
       │           │
       └─────┬─────┘
             ▼
Simulation Runtime / Deployment Runtime
             │
             ▼
Experiment registry → Reports → Reproduction bundle
```

Architectural rules:

- Data preparation cannot import Flower or training modules.
- Algorithm logic cannot depend directly on Ray, Docker, or deployment topology.
- Flower messages use versioned, validated payload schemas.
- Each client owns its private data, models, optimizer state policy, and checkpoints.
- The server owns only public open data, test data, aggregation state, and server checkpoints.
- Paper-reproduction behavior and later extensions must use separate named configurations.

---

## 3. Expanded Project Layout

```text
pyproject.toml
uv.lock
README.md
REPRODUCIBILITY.md
SECURITY.md
DATA_CARD.md
MODEL_CARD.md
CHANGELOG.md

configs/
  paper.yaml
  smoke.yaml
  deployment.yaml
  experiments.yaml
  robustness.yaml

src/ssfl/
  config/
  data/
  models/
  training/
  protocols/
  strategies/
  flower/
  experiments/
  evaluation/
  reporting/
  observability/
  security/
  deployment/

tests/
  unit/
  property/
  protocol/
  privacy/
  integration/
  deployment/
  performance/
  regression/

deployment/
  docker/
  compose/
  kubernetes/
  tls/
  scripts/

artifacts/
  data/
  runs/
  reports/
  checkpoints/
  bundles/
```

---

## 4. Milestone M0 — Requirements and Environment Gate

Before implementation:

- Confirm the actual repository path and Git state.
- Confirm where the 89 CSV files and `device_info.csv` are mounted.
- Verify the paper PDF checksum and record its citation.
- Inspect available CPU, RAM, GPU, CUDA, and disk capacity.
- Resolve current installable Python, PyTorch, Flower, Ray, Pandas, PyArrow, and scikit-learn versions.
- Record the exact Flower Message API and custom strategy interfaces used.
- Produce an implementation assumptions register.

The register must cover:

- Batch size 80 versus 100.
- CNN fully connected layer discrepancy.
- Normalization fitting scope.
- DS-FL temperature.
- FD loss composition.
- Optimizer-state persistence.
- Server-model initialization.
- No-vote and tied-vote behavior.
- Unspecified MLP/LSTM architectures.
- Meaning of a communication “round” in a two-phase SSFL exchange.

Gate: implementation begins only after these decisions are represented in configuration and `REPRODUCIBILITY.md`.

---

## 5. Milestone M1 — Foundation and Configuration

Implement:

- Python 3.12 package managed by `uv`.
- Modern Flower Message API application manifest.
- Pydantic configuration models.
- YAML experiment profiles.
- Structured logging.
- Deterministic seed utilities.
- Device selection supporting CPU, CUDA, and Apple MPS.
- Run-directory and artifact conventions.
- Configuration hashing.
- Environment snapshot capture.

Profiles:

- `smoke`: generated tiny dataset, three clients, two rounds.
- `paper`: complete paper settings.
- `paper_batch100`: reproduces the textual batch-size setting.
- `robustness`: multiple seeds and additional failure conditions.
- `deployment`: real network runtime.
- `debug`: single-process execution with extra assertions.

Gate:

- All profiles validate.
- Invalid combinations fail before training.
- Every run receives a stable run ID from configuration, dataset, code, and seed hashes.

---

## 6. Milestone M2 — Data Engineering and Governance

### Discovery

Support:

- Confirmed flat `<device>.<family>.<attack>.csv` naming.
- Optional nested official UCI layout.
- ZIP/RAR input archives.
- `device_info.csv`.
- Configurable but validated filename aliases.

### Validation

Validate:

- Nine device IDs.
- Devices 3 and 7 with six classes.
- Remaining seven devices with eleven classes.
- Exactly 115 identically ordered numeric features.
- Unique column names.
- No missing or infinite values.
- Minimum row count for every expected subset.
- No unexpected labels unless explicitly mapped.

### Sampling and splitting

- Deterministically sample 1,000 records per device/class.
- Store source file and original row number.
- Produce 89,000 total records.
- Split each device/class independently into 700/100/200.
- Assert 62,300 private, 8,900 open, and 17,800 test records.
- Guarantee no duplicate or cross-split source rows.
- Remove open labels from every training-accessible artifact.

### Transformation

- Support `all_mini` and `private_only` scaler fitting.
- Canonical paper profile uses `all_mini`.
- Store min, max, constant-feature mask, schema hash, and fit scope.
- Reshape 115 features to `(23,5)` exactly as Equation 19.
- Preserve both normalized flat and reshaped representations for model comparisons.

### Scenario generation

Produce immutable client manifests for:

- Scenario 1: 27 clients, two label-sorted shards each.
- Scenario 2: 89 clients, two shards each.
- Scenario 3: 89 clients, per-device Dirichlet allocation with `α=0.1`.

Also generate:

- Per-client class histograms.
- Jensen-Shannon divergence from the global distribution.
- Effective number of classes per client.
- Sample-count inequality statistics.
- Figure 2-style allocation plots.

### Data governance

- Dataset manifest and SHA-256 checksums.
- Data card with source, license, transformations, limitations, and leakage risks.
- Audit labels stored separately and inaccessible to training imports.
- Validation-only command that never rewrites artifacts.
- Atomic preparation: incomplete outputs never replace a valid prepared dataset.

Gate:

- All counts, hashes, disjointness, partition invariants, and Equation 19 tests pass.
- Re-running with the same inputs and seed produces identical manifests.

---

## 7. Milestone M3 — Models and Training Core

Implement:

- Paper CNN classifier and discriminator.
- MLP classifier/discriminator.
- LSTM classifier/discriminator.
- Model factory and typed model configuration.
- Parameter-count and activation-shape reports.
- Shared supervised and distillation training loops.
- Evaluation independent of Flower.
- Batched prediction over open data.
- Gradient clipping as an optional non-paper extension, disabled by default.
- Mixed precision as a performance profile, disabled for canonical reproduction.

Training policies:

- Adam, learning rate `1e-4`.
- Five epochs per configured phase.
- Canonical batch size 80.
- Cross-entropy for hard labels.
- Explicit teacher-distribution loss for soft labels.
- Fresh optimizer per Flower task while model weights persist.
- Deterministic DataLoader workers.
- No hidden dropout, augmentation, normalization layer, or scheduler in paper mode.

Gate:

- Exact CNN intermediate shapes.
- Correct output dimensions.
- Deterministic local training regression test.
- CPU, CUDA, and MPS paths produce valid outputs.

---

## 8. Milestone M4 — Federated Protocols

Define a versioned protocol state machine:

```text
INITIALIZED
  → LOCAL_SUPERVISED
  → PROPOSAL_UPLOADED
  → SERVER_AGGREGATED
  → GLOBAL_TARGET_BROADCAST
  → LOCAL_DISTILLED
  → EVALUATED
  → CHECKPOINTED
```

### SSFL

Implement:

- Local classifier training.
- Median-confidence calculation.
- Familiar/unfamiliar discriminator dataset.
- Discriminator training.
- Hard-label filtering.
- `-1` abstention.
- Server majority vote.
- Valid global-label mask.
- Client and server distillation.
- Persistent private client models.

### FL

Implement sample-weighted FedAvg with:

- Local training.
- Model upload.
- Weighted aggregation.
- Global broadcast.
- Server evaluation.

### FD

Implement:

- Per-class average client logits.
- Presence/count masks.
- Leave-self-out teacher calculation.
- Missing-class handling.
- Ground-truth and teacher losses.

### DS-FL

Implement:

- Full open-data soft predictions.
- Arithmetic aggregation.
- Temperature sharpening.
- Client and server distillation.

### Protocol safeguards

- Validate phase, round, sender, payload type, shape, dtype, dataset hash, and protocol version.
- Reject stale, duplicate, future-round, or wrong-algorithm messages.
- Make aggregation idempotent.
- Define retry-safe message IDs.
- Enforce configurable timeouts and minimum-client policies.
- Record abstentions, ties, missing classes, rejected messages, and failures.

Gate:

- Independent protocol tests pass without starting Flower.
- SSFL messages contain no client model parameters, gradients, private records, or optimizer state.

---

## 9. Milestone M5 — Flower Runtime Integration

### Flower application

Implement:

- Stateful `ClientApp`.
- `ServerApp`.
- Custom strategy lifecycle.
- Two exchanges per logical SSFL paper round.
- Array, metric, and configuration record adapters.
- Client-state serialization through `Context.state`.
- Server checkpoint and result serialization.

### Simulation

- Ray resource configuration.
- CPU-only mode.
- GPU client batching.
- Stable node-to-client-manifest mapping.
- Full participation in paper mode.
- Configurable client sampling only in extension profiles.

### Deployment

- SuperLink/server configuration.
- One SuperNode identity per logical client.
- Manifest-hash validation during startup.
- Client readiness barrier.
- Graceful shutdown.
- Reconnection and retry behavior.
- Checkpoint-based server restart.
- Client-local state recovery.

Gate:

- Two-round SSFL, FL, FD, and DS-FL simulations complete.
- Identical learning components work in separate-process deployment.
- Client identity cannot access another client partition.

---

## 10. Milestone M6 — Communication and Systems Measurement

Measure three layers:

1. Logical tensors.
2. Flower-serialized payloads.
3. Optional observed transport bytes.

Track:

- One-time open-data distribution.
- Per-round uploads and downloads.
- Broadcast multiplier.
- Message metadata overhead.
- Retry overhead.
- Checkpoint size.
- Server and client memory use.
- CPU/GPU utilization.
- Phase wall time.
- Throughput in open examples per second.
- Peak concurrent clients.

Generate:

- Per-client communication.
- Federation-wide communication.
- Cumulative cost curves.
- `C@50`, `C@75`, and `C@Top-Acc`.
- Logical versus serialized comparison.
- Communication versus accuracy Pareto plots.

Performance studies:

- 3, 27, and 89 simulated clients.
- Single and multiple Ray workers.
- CPU and available GPU.
- Different open-data batch sizes.
- Serialization cost of hard versus soft labels.
- State checkpoint overhead.

Gate:

- Byte accounting reconstructs every recorded message.
- SSFL hard labels are demonstrably smaller than DS-FL soft predictions.

---

## 11. Milestone M7 — Paper Experiments

### Main matrix

For every scenario:

- FL-CNN.
- FD-CNN.
- DS-FL-CNN.
- SSFL-CNN.
- SSFL-MLP.
- SSFL-LSTM.

Record rounds 10, 50, 100, 150, and 200.

### Ablations

- Full SSFL.
- No discriminator.
- No voting.
- No discriminator or voting.
- Simple confidence filtering.

### Threshold study

- 0.7.
- 0.8.
- 0.9.
- Per-client median.

### Label representation study

- Hard `int8`.
- Soft rounded to 2, 4, 6, and 8 decimal places.

### Additional robustness experiments

Keep separate from paper tables:

- Partial client availability.
- Slow clients.
- Client dropout between protocol phases.
- Corrupt or malformed messages.
- No-vote open samples.
- Extreme class imbalance.
- Noisy local labels.
- Poisoned pseudo-label client.
- Byzantine label proposals.
- Open-data distribution shift.
- Alternative Dirichlet α values: 0.01, 0.5, and 1.0.

Gate:

- Paper runs and extension runs remain clearly separated.
- Extension settings cannot silently alter canonical paper results.

---

## 12. Milestone M8 — Privacy and Security

Create an explicit threat model covering:

- Honest-but-curious server.
- Curious clients.
- Network observer.
- Compromised client.
- Malicious pseudo-label contributor.
- Model and membership inference attempts.
- Replay and stale-message attacks.

Verify:

- SSFL does not transmit private examples, gradients, or parameters.
- Hard labels can still leak model behavior; documentation must avoid claiming complete privacy.
- Audit artifacts are never server-readable in deployment.
- TLS for transport.
- Node authentication.
- Certificate generation and rotation instructions.
- Secret injection through environment or mounted files.
- No credentials committed to the repository.
- Payload-size and shape limits.
- Deserialization allowlist.
- Dependency and container vulnerability scanning.

Optional extension profiles:

- Differential privacy.
- Minimum vote quorum.
- Robust aggregation.
- Client reputation.
- Secure aggregation where applicable.

These extensions remain disabled in paper mode.

---

## 13. Milestone M9 — Reliability and Observability

### Logging

Use structured logs containing:

- Run ID.
- Client ID.
- Algorithm.
- Scenario.
- Logical round.
- Protocol phase.
- Message ID.
- Dataset/config hashes.
- Duration and result status.

Never log:

- Private features.
- Private labels.
- Complete local predictions tied to private records.
- Model tensors.
- Authentication secrets.

### Metrics

Expose or export:

- Active/ready/failed clients.
- Round and phase latency.
- Rejected/stale messages.
- Client abstention rates.
- Valid pseudo-label rate.
- Per-class vote distribution.
- Test metrics.
- Communication bytes.
- Memory and compute utilization.

### Recovery

- Atomic checkpoints.
- Last-completed-round marker.
- Resume only from phase-consistent state.
- Dataset/config/code compatibility validation.
- Configurable retry and timeout limits.
- Explicit behavior for insufficient clients.
- Run cancellation that preserves completed results.

---

## 14. Milestone M10 — Reporting and Scientific Reproducibility

Generate:

- Paper-style Tables II-IV.
- Figures 2-6.
- Per-seed and aggregate tables.
- Mean, standard deviation, confidence intervals.
- Per-class metrics.
- Raw and normalized confusion matrices.
- Communication-efficiency report.
- Runtime and resource report.
- Privacy/security audit report.
- Assumption and deviation report.

Every run directory contains:

```text
resolved_config.yaml
environment.json
dataset_manifest.json
code_version.json
metrics.parquet
communication.parquet
events.jsonl
checkpoints/
plots/
summary.json
```

Create a portable reproduction bundle containing:

- Configuration.
- Manifests and hashes.
- Metrics and plots.
- Dependency lock.
- Execution commands.
- Logs with sensitive data removed.
- Report.
- No raw private dataset.

---

## 15. Milestone M11 — Testing and CI

### Test layers

- Unit tests.
- Property-based partition and aggregation tests.
- Protocol state-machine tests.
- Security and privacy boundary tests.
- Integration simulations.
- Separate-process deployment test.
- Determinism regression.
- Performance smoke test.
- Report-generation snapshot tests.

### CI stages

1. Formatting and linting.
2. Static typing.
3. Dependency validation.
4. Unit and property tests.
5. CPU Flower smoke simulations.
6. Data-pipeline test with generated N-BaIoT-shaped CSVs.
7. Report-generation test.
8. Container build and vulnerability scan.
9. Optional scheduled GPU test.

Generated smoke data must reproduce:

- Nine devices.
- Devices 3 and 7 missing Mirai classes.
- Eleven global labels.
- 115 features.
- Imbalanced row counts.
- At least one constant feature.
- Edge cases for NaN, insufficient rows, and schema mismatch.

---

## 16. Milestone M12 — Packaging, Documentation, and Handoff

Documentation must include:

- Installation for CPU, CUDA, and MPS.
- `/data` input contract.
- Preparation command.
- Simulation commands.
- Full-suite commands.
- Deployment instructions.
- Resume and recovery procedures.
- Artifact interpretation.
- Paper-versus-reproduction comparison.
- Known limitations.
- Security guidance.
- Hardware sizing.
- Troubleshooting matrix.

Deployment deliverables:

- CPU Docker image.
- GPU Docker image.
- Docker Compose smoke topology.
- Generated 27/89-client launch configurations.
- Kubernetes manifests or Helm values.
- TLS example.
- Health checks.
- Persistent volume definitions.
- Resource requests and limits.

---

## 17. Verification Gates

### Gate A — Foundation

- Dependency resolution succeeds.
- Profiles validate.
- Environment snapshot is generated.

### Gate B — Data

- Real dataset produces exactly 89,000 selected records.
- Split totals and client counts match.
- Checksums and validation rerun pass.

### Gate C — Learning core

- Model shapes and local training tests pass.
- Protocols pass independently of Flower.

### Gate D — CPU smoke

- Two rounds of SSFL, FL, FD, and DS-FL complete.
- No prohibited SSFL payloads are present.
- Resume reproduces uninterrupted results.

### Gate E — Reporting

- Smoke matrix generates all table and figure formats.
- Communication totals reconcile with messages.

### Gate F — Deployment

- One server and three separate clients complete a run.
- TLS/authentication profile starts successfully.
- Wrong manifests and stale messages are rejected.

### Gate G — Full-run readiness

- Paper configuration expands to the complete experiment matrix.
- GPU resource estimate and expected runtime are documented.
- Launch and resume commands are validated without starting the expensive run.

---

## 18. Final Scope Boundary

Included:

- Complete paper reproduction code.
- Smoke verification.
- Full experiment configurations.
- Research extensions.
- Simulation and deployment.
- Security, observability, reporting, CI, and documentation.

Configured but not executed during the build:

- Full 200-round paper matrix.
- Full three-seed GPU suite.
- Large-scale Kubernetes deployment.
- Optional differential privacy and Byzantine-defense studies.

These require user-provided data, GPU capacity, and—in the case of production deployment—certificates and infrastructure credentials.
