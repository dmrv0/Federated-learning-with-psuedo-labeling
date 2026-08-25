import numpy as np
import pytest

from ssfl.protocols.message import ProtocolError
from ssfl.protocols.payload_limits import (
    validate_dsfl_arrays,
    validate_fd_arrays,
    validate_ssfl_proposal_arrays,
)

NUM_CLASSES = 11
NUM_OPEN = 4


def test_ssfl_hard_labels_accepted():
    arrays = {"pseudo_labels": np.array([2, -1, 0, 10], dtype=np.int8)}
    validate_ssfl_proposal_arrays(
        arrays, num_open=NUM_OPEN, num_classes=NUM_CLASSES
    )  # must not raise


def test_ssfl_soft_probs_accepted():
    row = np.zeros(NUM_CLASSES, dtype=np.float32)
    row[0] = 1.0
    soft = np.stack([row, np.zeros(NUM_CLASSES, dtype=np.float32), row, row])
    arrays = {"soft_probs": soft.astype(np.float32)}
    validate_ssfl_proposal_arrays(
        arrays, num_open=NUM_OPEN, num_classes=NUM_CLASSES
    )  # must not raise


def test_ssfl_rejects_wrong_shape():
    arrays = {"pseudo_labels": np.array([2, -1], dtype=np.int8)}
    with pytest.raises(ProtocolError):
        validate_ssfl_proposal_arrays(arrays, num_open=NUM_OPEN, num_classes=NUM_CLASSES)


def test_ssfl_rejects_noncanonical_confidence_upload():
    arrays = {
        "confidences": np.array([0.9, 0.5, 0.4, 0.1], dtype=np.float32),
        "pseudo_labels": np.array([2, -1, 0, 1], dtype=np.int8),
    }
    with pytest.raises(ProtocolError):
        validate_ssfl_proposal_arrays(arrays, num_open=NUM_OPEN, num_classes=NUM_CLASSES)


def test_ssfl_rejects_out_of_range_pseudo_label():
    arrays = {"pseudo_labels": np.array([2, -1, 99, 1], dtype=np.int16)}
    with pytest.raises(ProtocolError):
        validate_ssfl_proposal_arrays(arrays, num_open=NUM_OPEN, num_classes=NUM_CLASSES)


def test_ssfl_rejects_both_or_neither_label_representation():
    labels = np.array([2, -1, 0, 1], dtype=np.int8)
    soft = np.zeros((NUM_OPEN, NUM_CLASSES), dtype=np.float32)
    with pytest.raises(ProtocolError):
        validate_ssfl_proposal_arrays(
            {"pseudo_labels": labels, "soft_probs": soft},
            num_open=NUM_OPEN,
            num_classes=NUM_CLASSES,
        )
    with pytest.raises(ProtocolError):
        validate_ssfl_proposal_arrays({}, num_open=NUM_OPEN, num_classes=NUM_CLASSES)


def test_ssfl_rejects_nan():
    arrays = {"soft_probs": np.full((NUM_OPEN, NUM_CLASSES), np.nan, dtype=np.float32)}
    with pytest.raises(ProtocolError):
        validate_ssfl_proposal_arrays(arrays, num_open=NUM_OPEN, num_classes=NUM_CLASSES)


def _fd_arrays(present_classes):
    class_probs = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float32)
    class_present = np.zeros(NUM_CLASSES, dtype=np.int32)
    for c in present_classes:
        class_probs[c, c] = 1.0
        class_present[c] = 1
    return {"class_probs": class_probs, "class_present": class_present}


def test_fd_accepts_valid_upload():
    validate_fd_arrays(_fd_arrays([0, 3, 7]), num_classes=NUM_CLASSES)  # must not raise


def test_fd_rejects_row_not_summing_to_one():
    arrays = _fd_arrays([0])
    arrays["class_probs"][0, 0] = 0.4  # present class row no longer sums to ~1
    with pytest.raises(ProtocolError):
        validate_fd_arrays(arrays, num_classes=NUM_CLASSES)


def test_fd_rejects_non_binary_class_present():
    arrays = _fd_arrays([0])
    arrays["class_present"][1] = 5
    with pytest.raises(ProtocolError):
        validate_fd_arrays(arrays, num_classes=NUM_CLASSES)


def test_fd_rejects_wrong_shape():
    arrays = _fd_arrays([0])
    arrays["class_probs"] = arrays["class_probs"][:, :5]
    with pytest.raises(ProtocolError):
        validate_fd_arrays(arrays, num_classes=NUM_CLASSES)


def test_dsfl_accepts_row_stochastic_probs():
    probs = np.full((NUM_OPEN, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float32)
    validate_dsfl_arrays(
        {"probs": probs}, num_open=NUM_OPEN, num_classes=NUM_CLASSES
    )  # must not raise


def test_dsfl_rejects_row_not_summing_to_one():
    probs = np.zeros((NUM_OPEN, NUM_CLASSES), dtype=np.float32)
    probs[:, 0] = 0.5
    with pytest.raises(ProtocolError):
        validate_dsfl_arrays({"probs": probs}, num_open=NUM_OPEN, num_classes=NUM_CLASSES)


def test_dsfl_rejects_integer_dtype():
    probs = np.zeros((NUM_OPEN, NUM_CLASSES), dtype=np.int32)
    with pytest.raises(ProtocolError):
        validate_dsfl_arrays({"probs": probs}, num_open=NUM_OPEN, num_classes=NUM_CLASSES)
