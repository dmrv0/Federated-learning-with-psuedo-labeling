"""Fast, no-subprocess checks for generate_launch_configs.py's flag branching -- the real
process-level wiring is covered by test_deployment_smoke.py (marked slow)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deployment.generate_launch_configs import generate


def _write_manifest(data_path: Path, scenario: int, num_clients: int) -> None:
    scenarios_dir = data_path / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    payload = {"clients": [{"client_id": f"c{i}"} for i in range(num_clients)]}
    (scenarios_dir / f"{scenario}.json").write_text(json.dumps(payload))


def test_insecure_profile_omits_tls_flags(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "data", scenario=1, num_clients=5)
    out = tmp_path / "out"
    generate(
        data_path=tmp_path / "data",
        scenario=1,
        profile="insecure",
        output_dir=out,
        fleet_address="127.0.0.1:9092",
        control_address="127.0.0.1:9093",
        serverappio_address="127.0.0.1:9091",
        num_clients=None,
        certs_dir=tmp_path / "certs",
    )
    superlink = (out / "superlink.sh").read_text()
    supernodes = (out / "supernodes.sh").read_text()
    assert "--insecure" in superlink
    assert "--ssl-certfile" not in superlink
    assert supernodes.count("--insecure") == 5
    assert "root-certificates" not in supernodes
    assert (out / "flwr_config_snippet.toml").read_text().count("insecure = true") == 1


def test_tls_profile_uses_quoted_cert_paths(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "data", scenario=1, num_clients=2)
    out = tmp_path / "out"
    certs_dir = tmp_path / "certs with spaces"
    generate(
        data_path=tmp_path / "data",
        scenario=1,
        profile="tls",
        output_dir=out,
        fleet_address="127.0.0.1:9092",
        control_address="127.0.0.1:9093",
        serverappio_address="127.0.0.1:9091",
        num_clients=None,
        certs_dir=certs_dir,
    )
    superlink = (out / "superlink.sh").read_text()
    supernodes = (out / "supernodes.sh").read_text()
    assert "--insecure" not in superlink
    assert f"'{certs_dir}/server.pem'" in superlink
    assert f"'{certs_dir}/ca.crt'" in supernodes
    assert "root-certificates" in (out / "flwr_config_snippet.toml").read_text()


def test_num_clients_overflow_rejected(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "data", scenario=1, num_clients=3)
    with pytest.raises(ValueError, match="exceeds"):
        generate(
            data_path=tmp_path / "data",
            scenario=1,
            profile="insecure",
            output_dir=tmp_path / "out",
            fleet_address="127.0.0.1:9092",
            control_address="127.0.0.1:9093",
            serverappio_address="127.0.0.1:9091",
            num_clients=4,
            certs_dir=tmp_path / "certs",
        )


def test_supernode_clientappio_ports_are_distinct(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "data", scenario=1, num_clients=3)
    out = tmp_path / "out"
    generate(
        data_path=tmp_path / "data",
        scenario=1,
        profile="insecure",
        output_dir=out,
        fleet_address="127.0.0.1:9092",
        control_address="127.0.0.1:9093",
        serverappio_address="127.0.0.1:9091",
        num_clients=None,
        certs_dir=tmp_path / "certs",
    )
    supernodes = (out / "supernodes.sh").read_text()
    ports = {line.split("--clientappio-api-address ")[1].split()[0] for line in supernodes.splitlines() if "flower-supernode" in line}
    assert len(ports) == 3
