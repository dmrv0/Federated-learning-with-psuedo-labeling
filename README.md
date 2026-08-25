# SSFL — Semisupervised Federated Learning for IoT Intrusion Detection

Reproduction of *"Semisupervised Federated-Learning-Based Intrusion Detection Method for Internet
of Things"* on N-BaIoT: four federated protocols (SSFL, FL, FD, DS-FL) across three non-IID client
scenarios, built on Flower + PyTorch. See `SSFL_IMPLEMENTATION_PLAN.md` for the full spec this repo
implements, and `REPRODUCIBILITY.md` for every deviation from the paper and why.

## Layout

```
src/ssfl/
  config.py                pydantic profile config (smoke/paper/deployment/...)
  data/                     prepare_data, discovery, partition, datasets, scaling
  models.py  training.py    CNN/MLP/LSTM backbones, train/eval loops
  protocols/                per-algorithm aggregation logic (ssfl, fl, fd, dsfl) + message contract
  strategies/                per-algorithm Flower Strategy
  client_app.py  server_app.py
  experiments/run_suite.py  reporting/build_report.py
deployment/                 real (non-simulation) SuperLink/SuperNode launch scripts + TLS certs
configs/                    smoke / paper / deployment / experiments*.yaml profiles
tests/                      unit, protocol/security, integration, deployment
artifacts/                  prepared data + run outputs (gitignored)
```

## Quickstart (CPU smoke)

```bash
uv sync

uv run python -m ssfl.data.prepare_data --input data --output artifacts/data --seed 2023

uv run pytest -q                    # unit + protocol/security tests (~45s)
uv run pytest -m slow -q            # + real flwr simulations and deployment processes (minutes)

uv run flwr run . \
  --run-config 'profile="smoke" algorithm="ssfl" scenario=1 device="cpu"' \
  --federation-config 'num-supernodes=27 client-resources-num-cpus=1 client-resources-num-gpus=0 init-args-num-cpus=8' \
  --stream

uv run python -m ssfl.experiments.run_suite --matrix configs/experiments_smoke.yaml
uv run python -m ssfl.reporting.build_report --runs artifacts/runs --output artifacts/report
```

`algorithm` is one of `ssfl`/`fl`/`fd`/`dsfl`; `scenario` is `1` (27 clients), `2`, or `3` (89
clients each).

## RTX 3090 paper runs

The canonical profile uses CUDA, deterministic kernels, 200 rounds, 5 local epochs, Adam at
`1e-4`, batch size 80, checkpoints every round, and eight concurrent actors at `0.125` GPU each:

```bash
uv run flwr run . \
  --run-config 'profile="paper" algorithm="ssfl" scenario=1 device="cuda"' \
  --federation-config 'num-supernodes=27 client-resources-num-cpus=1 client-resources-num-gpus=0.125 init-args-num-cpus=8 init-args-num-gpus=1' \
  --stream
```

For scenarios 2/3 use `scenario=2` or `3` and `num-supernodes=89`. To execute the full paper
matrix—including all three scenarios for every ablation, threshold, and label-representation
study—run:

```bash
uv run python -m ssfl.experiments.run_suite --matrix configs/experiments.yaml --resume
```

To run the proposed SSFL-CNN solution for scenarios 1, 2, and 3 first, use
`configs/experiments_solution.yaml`. The entries have the same identities as the full matrix, so a
later `configs/experiments.yaml --resume` run skips those completed results.
`scripts/run_solution_with_alert.sh` is a background-friendly launcher that records completion in
`artifacts/logs/solution_scenarios.status` and sends a Linux desktop notification/sound when
available.

To resume one interrupted run from its last completed round, add
`resume-from="artifacts/runs/<run-id>"` to `--run-config` while keeping the original profile,
algorithm, scenario, device, and federation configuration.

## Exhaustive artifacts

Every epoch, client phase, aggregation phase, communication round, evaluation, checkpoint action,
and one-second GPU/system sample is flushed during paper runs. Per-mini-batch events are disabled
to keep the full matrix within practical storage limits; set `log_every_batch: true` only for a
short diagnostic profile. Each deterministic run directory has:

```text
metrics.parquet                 per-round macro/micro/weighted metrics
per_class_metrics.parquet       per-class precision/recall/F1/support
confusion_matrices.npz          raw confusion matrix per round
communication.parquet          every message, tensor shape/dtype, real/wire/paper bytes
checkpoints/                    server milestones/final and latest resumable per-client weights
attempts/<attempt-id>/
  events.jsonl[.gz]             compact server aggregation stream
  telemetry/server.jsonl[.gz]   rounds, phases, GPU/system, evaluation details
  telemetry/clients/*.jsonl[.gz] every client epoch/phase/prediction/evaluation summary
  aggregation_audit/*.npz       full SSFL vote counts, participation, labels, masks
```

Raw private samples, model weights, gradients, and secret values are deliberately not copied into
JSON logs. The matrix runner losslessly gzip-compresses JSONL after each run completes; active and
interrupted runs stay uncompressed for live tailing. Checkpoints contain model state and must be
protected as sensitive artifacts.

## Real (non-simulation) deployment

`flwr run` above uses Flower's local Simulation/Ray backend. `deployment/` has a separate scaffold
for real SuperLink + SuperNode processes (insecure-dev and TLS profiles) — see `deployment/README.md`.

## Docs

- `REPRODUCIBILITY.md` — every assumption, deviation, and resolved ambiguity vs. the paper, numbered.
- `DATA_CARD.md` — N-BaIoT source, transformations, splits, leakage mitigations.
- `MODEL_CARD.md` — backbones, intended use, evaluation, limitations.
- `SECURITY.md` — threat model, verified privacy boundaries per protocol, deployment gap notes.
