# Deployment

Real (non-simulation) SuperLink + SuperNode processes, as an alternative to the Flower
Simulation/Ray backend `flwr run` uses everywhere else in this repo. Same `ClientApp`/`ServerApp`,
same `configs/deployment.yaml` (200-round paper profile). See `SECURITY.md`'s "Deployment gap"
section for exactly what this scaffold does and doesn't cover.

## Generate launch scripts

```bash
uv run python deployment/generate_launch_configs.py --scenario 1 --profile insecure
uv run python deployment/generate_launch_configs.py --scenario 1 --profile tls --num-clients 3
```

Writes `superlink.sh`, `supernodes.sh` (one `flower-supernode` line per client, backgrounded), and
`flwr_config_snippet.toml` into `--output-dir` (default `deployment/generated/`, gitignored). Client
count comes from the scenario manifest (`artifacts/data/scenarios/<n>.json`) — `--num-clients` caps
it to a subset (e.g. for a smoke test); omit it to launch the scenario's full 27 (scenario 1) or 89
(scenario 2/3).

## `tls` profile: generate dev certs first

```bash
deployment/certs/generate_dev_certs.sh
```

Self-signed CA + SuperLink leaf cert (`ca.crt`/`ca.key`/`server.pem`/`server.key`, gitignored) —
fine for a local/dev deployment, not a real PKI. A production deployment needs certs from a real CA.

## Run it

```bash
bash deployment/generated/<output-dir>/superlink.sh    # terminal 1
bash deployment/generated/<output-dir>/supernodes.sh    # terminal 2
```

Then, for the `insecure` profile, append `flwr_config_snippet.toml`'s contents to
`$HOME/.flwr/config.toml` and run:

```bash
flwr run . ssfl-insecure
```

## Verifying it works

`tests/deployment/test_deployment_smoke.py` (`pytest -m slow tests/deployment`) spawns a real
SuperLink + 3 SuperNodes as subprocesses for both profiles and asserts clean registration
(`Fleet.ActivateNode`) and shutdown. It does **not** run a full federated round — see
`SECURITY.md`'s "Deployment gap" for why a literal full round with only 3 clients is structurally
impossible (fixed per-scenario client counts, no partial-participation support).

`tests/deployment/test_generate_launch_configs.py` is a fast (non-`slow`) unit test of the
generator's flag branching — no real processes.

## Docker/Compose (written, not live-verified here)

`docker-compose.yml` builds the same insecure-profile topology (1 SuperLink + 3 SuperNodes) as
containers instead of native processes:

```bash
docker compose -f deployment/docker-compose.yml build
docker compose -f deployment/docker-compose.yml up
```

**Not run in this repo's build environment** — see `REPRODUCIBILITY.md` #30: building the image
(torch/ray/flwr inside `python:3.12-slim`) filled this machine's disk to 99% via colima's VM disk.
Check `df -h` for real headroom (15GB+) before building. Each container gets its own network
namespace, so — unlike co-located native processes — the SuperNodes don't need distinct
`--clientappio-api-address` values.
