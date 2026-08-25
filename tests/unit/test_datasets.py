import csv
from pathlib import Path

import numpy as np
import pytest

from ssfl.config import Backbone, DataPrepConfig
from ssfl.data.datasets import load_client_assignments, load_client_private_data, load_open_data, load_test_data
from ssfl.data.labels import LABEL_MAP, NUM_FEATURES
from ssfl.data.prepare_data import run_full

FEATURE_NAMES = [f"f{i}" for i in range(NUM_FEATURES)]
SIX_CLASS_DEVICES = {3, 7}


def _write_csv(path: Path, num_rows: int, rng: np.random.Generator) -> None:
    data = rng.random((num_rows, NUM_FEATURES), dtype=np.float64) * 100
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(FEATURE_NAMES)
        writer.writerows(data.tolist())


def _make_synthetic_dataset(root: Path, rows_per_file: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    with open(root / "device_info.csv", "w") as fh:
        fh.write("DeviceID,DeviceName\n")
        for d in range(1, 10):
            fh.write(f"{d},Device{d}\n")
    for device_id in range(1, 10):
        keys = [k for k in LABEL_MAP if k.startswith("gafgyt.") or k == "benign"]
        if device_id not in SIX_CLASS_DEVICES:
            keys += [k for k in LABEL_MAP if k.startswith("mirai.")]
        for class_key in keys:
            _write_csv(root / f"{device_id}.{class_key}.csv", rows_per_file, rng)


@pytest.fixture(scope="module")
def prepared_data_root(tmp_path_factory) -> Path:
    raw = tmp_path_factory.mktemp("raw")
    _make_synthetic_dataset(raw, rows_per_file=60)
    output = tmp_path_factory.mktemp("out") / "out"
    config = DataPrepConfig(input_path=raw, output_path=output, seed=2023, samples_per_subset=50)
    run_full(config)
    return output


def test_load_client_assignments_scenario_1(prepared_data_root: Path) -> None:
    assignments = load_client_assignments(prepared_data_root, scenario=1)
    assert len(assignments) == 27
    assert all(a.scenario == 1 for a in assignments)


@pytest.mark.parametrize("backbone", list(Backbone))
def test_load_client_private_data_matches_assignment(prepared_data_root: Path, backbone: Backbone) -> None:
    assignments = load_client_assignments(prepared_data_root, scenario=1)
    assignment = assignments[0]
    dataset = load_client_private_data(prepared_data_root, assignment, backbone)
    assert len(dataset) == assignment.num_examples
    x0 = dataset[0][0]
    if backbone == Backbone.mlp:
        assert x0.shape == (115,)
    else:
        assert x0.shape == (23, 5)
    labels_seen = {int(dataset[i][1]) for i in range(len(dataset))}
    assert labels_seen == set(assignment.class_local_indices.keys())


def test_load_open_data_has_no_labels(prepared_data_root: Path) -> None:
    dataset = load_open_data(prepared_data_root, Backbone.cnn)
    assert dataset.y is None
    assert len(dataset) == 89 * 5  # 50 samples/subset * 0.1 open ratio


def test_load_test_data_has_labels(prepared_data_root: Path) -> None:
    dataset = load_test_data(prepared_data_root, Backbone.cnn)
    assert dataset.y is not None
    assert len(dataset) == 89 * 10  # 50 samples/subset * 0.2 test ratio
    assert int(dataset[0][1]) in LABEL_MAP.values()
