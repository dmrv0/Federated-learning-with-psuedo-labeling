"""numpy <-> ``ArrayRecord`` adapters shared by ``client_app.py`` and ``strategies/*.py``.

Model weights use ``ArrayRecord.from_torch_state_dict``/``.to_torch_state_dict`` directly at the
call site (no wrapper needed); this module only covers the plain-numpy payloads that SSFL/FD/DS-FL
carry (pseudo-labels, class logits, soft predictions -- never model parameters).
"""

from __future__ import annotations

import numpy as np
from flwr.common import Array, ArrayRecord


def array_record_from_numpy(arrays: dict[str, np.ndarray]) -> ArrayRecord:
    return ArrayRecord(array_dict={key: Array.from_numpy_ndarray(value) for key, value in arrays.items()})


def numpy_from_array_record(record: ArrayRecord) -> dict[str, np.ndarray]:
    return {key: array.numpy() for key, array in record.items()}


if __name__ == "__main__":
    _roundtrip = numpy_from_array_record(array_record_from_numpy({"a": np.arange(5, dtype=np.int64)}))
    assert np.array_equal(_roundtrip["a"], np.arange(5, dtype=np.int64))
    print("records.py self-check OK")
