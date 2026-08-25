"""Emit shell launch scripts for a real (non-simulation) SuperLink + SuperNode deployment.

Client count comes straight from the scenario manifest (``artifacts/data/scenarios/<n>.json``) --
``_partition_client`` in ``client_app.py`` indexes ``assignments[partition_id]`` positionally, so
SuperNode ``i`` just needs ``partition-id=i num-partitions=<len(assignments)>``; no per-client data
beyond the count is needed to generate a launch command.

Usage::

    uv run python deployment/generate_launch_configs.py --scenario 1 --profile insecure
    uv run python deployment/generate_launch_configs.py --scenario 1 --profile tls \\
        --num-clients 3  # smoke: launch only a subset of the scenario's full client count
"""

from __future__ import annotations

import argparse
import json
import shlex
import stat
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _manifest_client_count(data_path: Path, scenario: int) -> int:
    payload = json.loads((data_path / "scenarios" / f"{scenario}.json").read_text())
    return len(payload["clients"])


def _write_script(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def generate(
    data_path: Path,
    scenario: int,
    profile: str,
    output_dir: Path,
    fleet_address: str,
    control_address: str,
    serverappio_address: str,
    num_clients: int | None,
    certs_dir: Path,
    clientappio_base_port: int = 9094,
    clientappio_base_port_host: str = "127.0.0.1",
) -> None:
    total_clients = _manifest_client_count(data_path, scenario)
    launch_count = num_clients if num_clients is not None else total_clients
    if launch_count > total_clients:
        raise ValueError(f"--num-clients {launch_count} exceeds scenario {scenario}'s {total_clients} clients")

    # shlex.quote: certs_dir / REPO_ROOT can contain spaces (this repo's dir name does).
    tls_flags_superlink = (
        "--insecure"
        if profile == "insecure"
        else f"--ssl-certfile {shlex.quote(str(certs_dir / 'server.pem'))} "
        f"--ssl-keyfile {shlex.quote(str(certs_dir / 'server.key'))} "
        f"--ssl-ca-certfile {shlex.quote(str(certs_dir / 'ca.crt'))}"
    )
    tls_flags_supernode = (
        "--insecure"
        if profile == "insecure"
        else f"--root-certificates {shlex.quote(str(certs_dir / 'ca.crt'))}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    superlink_script = f"""#!/usr/bin/env bash
# SuperLink -- {profile} profile, scenario {scenario} ({launch_count}/{total_clients} clients).
set -euo pipefail
exec flower-superlink {tls_flags_superlink} \\
  --fleet-api-address {fleet_address} \\
  --control-api-address {control_address} \\
  --serverappio-api-address {serverappio_address}
"""
    _write_script(output_dir / "superlink.sh", superlink_script)

    # Each SuperNode's ClientAppIo API defaults to 0.0.0.0:9094 -- fine for one node per machine,
    # but co-located SuperNodes (this smoke topology) collide on that port unless given distinct ones.
    clientappio_host = clientappio_base_port_host
    supernode_lines = ["#!/usr/bin/env bash", "# All SuperNodes for this topology -- run each line as a separate process.", "set -euo pipefail", ""]
    for partition_id in range(launch_count):
        clientappio_address = f"{clientappio_host}:{clientappio_base_port + partition_id}"
        supernode_lines.append(
            f"flower-supernode {tls_flags_supernode} --superlink {fleet_address} "
            f"--clientappio-api-address {clientappio_address} "
            f"--node-config 'partition-id={partition_id} num-partitions={total_clients}' &"
        )
    supernode_lines.append("wait")
    _write_script(output_dir / "supernodes.sh", "\n".join(supernode_lines) + "\n")

    flwr_config_name = f"ssfl-{profile}"
    if profile == "insecure":
        config_snippet = f'[superlink.{flwr_config_name}]\naddress = "{control_address}"\ninsecure = true\n'
    else:
        config_snippet = (
            f'[superlink.{flwr_config_name}]\naddress = "{control_address}"\n'
            f'root-certificates = "{certs_dir}/ca.crt"\n'
        )
    (output_dir / "flwr_config_snippet.toml").write_text(
        f"# Append to $HOME/.flwr/config.toml (or $FLWR_HOME/config.toml), then:\n"
        f"#   flwr run {REPO_ROOT} {flwr_config_name}\n{config_snippet}"
    )

    print(f"wrote {output_dir}/superlink.sh, supernodes.sh ({launch_count} nodes), flwr_config_snippet.toml")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--profile", choices=["insecure", "tls"], required=True)
    parser.add_argument("--data-path", type=Path, default=REPO_ROOT / "artifacts" / "data")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "deployment" / "generated")
    parser.add_argument("--certs-dir", type=Path, default=REPO_ROOT / "deployment" / "certs")
    parser.add_argument("--fleet-address", default="127.0.0.1:9092")
    parser.add_argument("--control-address", default="127.0.0.1:9093")
    parser.add_argument("--serverappio-address", default="127.0.0.1:9091")
    parser.add_argument(
        "--num-clients", type=int, default=None, help="Launch only a subset (e.g. for a smoke test); default = all"
    )
    args = parser.parse_args()

    generate(
        data_path=args.data_path,
        scenario=args.scenario,
        profile=args.profile,
        output_dir=args.output_dir,
        fleet_address=args.fleet_address,
        control_address=args.control_address,
        serverappio_address=args.serverappio_address,
        num_clients=args.num_clients,
        certs_dir=args.certs_dir,
    )


if __name__ == "__main__":
    main()
