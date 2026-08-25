"""Real separate-process deployment smoke test: a SuperLink + 3 SuperNodes as actual OS
processes, talking gRPC over loopback, using the generated launch scripts from
``deployment/generate_launch_configs.py``.

Scope, per REPRODUCIBILITY.md: ``ExperimentConfig.num_clients()`` is a fixed mapping
(scenario 1 = 27) and the custom strategies require an exact client-count match (no
partial-participation support), so a literal "3 clients complete a full federated round"
is structurally impossible. This test instead verifies process-level wiring -- SuperLink
starts, 3 SuperNodes register against it (``Fleet.ActivateNode``), then everything shuts
down cleanly -- for both the insecure-dev and TLS profiles.

Also: Flower's SuperNode registration handshake is transport-level and does not carry the
app's ``ConfigRecord``/``ArrayRecord`` payloads, so there is no "manifest-hash check at
registration" to observe here -- ``dataset_manifest_hash`` validation
(``protocols/message.py::validate_envelope``) only fires on the first real train/evaluate
message exchange, which requires a full client count and is out of scope for this smoke test.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from deployment.generate_launch_configs import generate

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "artifacts" / "data"
CERTS_DIR = REPO_ROOT / "deployment" / "certs"
NUM_SMOKE_CLIENTS = 3
REGISTER_TIMEOUT_S = 20


def _wait_for(predicate, timeout_s: float, poll_s: float = 0.5) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return predicate()


def _run_topology(profile: str, output_dir: Path, base_port: int) -> None:
    if profile == "tls" and not (CERTS_DIR / "ca.crt").exists():
        pytest.skip("deployment/certs/ca.crt missing -- run deployment/certs/generate_dev_certs.sh first")

    fleet_address = f"127.0.0.1:{base_port}"
    control_address = f"127.0.0.1:{base_port + 1}"
    serverappio_address = f"127.0.0.1:{base_port + 2}"
    generate(
        data_path=DATA_PATH,
        scenario=1,
        profile=profile,
        output_dir=output_dir,
        fleet_address=fleet_address,
        control_address=control_address,
        serverappio_address=serverappio_address,
        num_clients=NUM_SMOKE_CLIENTS,
        certs_dir=CERTS_DIR,
        clientappio_base_port=base_port + 3,
    )

    superlink_log = (output_dir / "superlink.out").open("w")
    supernodes_log = (output_dir / "supernodes.out").open("w")
    superlink_proc = subprocess.Popen(
        ["bash", str(output_dir / "superlink.sh")], stdout=superlink_log, stderr=subprocess.STDOUT
    )
    try:
        assert _wait_for(lambda: "Fleet API" in (output_dir / "superlink.out").read_text(), timeout_s=15), (
            output_dir / "superlink.out"
        ).read_text()

        supernodes_proc = subprocess.Popen(
            ["bash", str(output_dir / "supernodes.sh")], stdout=supernodes_log, stderr=subprocess.STDOUT
        )
        try:
            registered = _wait_for(
                lambda: (output_dir / "superlink.out").read_text().count("Fleet.ActivateNode") == NUM_SMOKE_CLIENTS,
                timeout_s=REGISTER_TIMEOUT_S,
            )
            superlink_text = (output_dir / "superlink.out").read_text()
            supernodes_text = (output_dir / "supernodes.out").read_text()
            assert registered, f"superlink:\n{superlink_text}\n\nsupernodes:\n{supernodes_text}"
            assert supernodes_text.count("SuperNode ID:") == NUM_SMOKE_CLIENTS
            assert "already in use" not in supernodes_text
            assert "Traceback" not in supernodes_text
        finally:
            supernodes_proc.terminate()
            supernodes_proc.wait(timeout=10)
    finally:
        superlink_proc.terminate()
        superlink_proc.wait(timeout=10)
        superlink_log.close()
        supernodes_log.close()


@pytest.mark.slow
def test_insecure_topology_registers_three_supernodes(tmp_path: Path) -> None:
    _run_topology("insecure", tmp_path / "insecure_smoke", base_port=19090)


@pytest.mark.slow
def test_tls_topology_registers_three_supernodes(tmp_path: Path) -> None:
    _run_topology("tls", tmp_path / "tls_smoke", base_port=19190)
