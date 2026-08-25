"""Shared fixtures for privacy/integration/deployment tests that need a real prepared-data tree
rather than fabricated arrays -- same synthetic-CSV approach as ``tests/unit/test_datasets.py``,
factored out so ``tests/privacy``/``tests/integration``/``tests/deployment`` don't each reinvent it.
"""

from pathlib import Path

import numpy as np
import pytest

from ssfl.config import DataPrepConfig
from ssfl.data.labels import LABEL_MAP, NUM_FEATURES
from ssfl.data.prepare_data import run_full

FEATURE_NAMES = [f"f{i}" for i in range(NUM_FEATURES)]
SIX_CLASS_DEVICES = {3, 7}


def _write_csv(path: Path, num_rows: int, rng: np.random.Generator) -> None:
    import csv

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


@pytest.fixture(scope="session")
def prepared_data_root(tmp_path_factory) -> Path:
    raw = tmp_path_factory.mktemp("raw")
    _make_synthetic_dataset(raw, rows_per_file=60)
    output = tmp_path_factory.mktemp("out") / "out"
    config = DataPrepConfig(input_path=raw, output_path=output, seed=2023, samples_per_subset=50)
    run_full(config)
    return output
