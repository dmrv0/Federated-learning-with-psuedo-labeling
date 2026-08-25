"""Client-reply payload validation: shape/dtype/range limits shared by all custom strategies.

Complements ``protocols/message.py``'s envelope checks (identity/replay/staleness) with the
payload-content checks M8/M9 call out separately ("payload-size and shape limits," "deserialization
allowlist," defense against a "malicious pseudo-label contributor"). A client that returns a
wrong-shape, wrong-dtype, NaN/Inf, or out-of-probability-range array would otherwise either crash
``aggregate_train`` (denial of service for the whole round) or silently corrupt the vote/average it
feeds into (a malicious client shouldn't be able to inject e.g. a pseudo-label >= num_classes and
skew ``votes[idx, labels[idx]]`` out of bounds). No ``flwr`` import, matching ``message.py``'s
Flower-independent scope -- callers pull arrays out of a ``Message`` first, then validate.
"""

from __future__ import annotations

import numpy as np

from ssfl.protocols.message import ProtocolError

_PROB_TOL = 1e-3


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def _check_shape(name: str, arr: np.ndarray, shape: tuple[int, ...]) -> None:
    _require(arr.shape == shape, f"{name}: expected shape {shape}, got {arr.shape}")


def _check_dtype_kind(name: str, arr: np.ndarray, kind: str) -> None:
    _require(arr.dtype.kind == kind, f"{name}: expected dtype kind {kind!r}, got {arr.dtype}")


def _check_finite(name: str, arr: np.ndarray) -> None:
    _require(bool(np.isfinite(arr).all()), f"{name}: contains NaN/Inf")


def _check_range(name: str, arr: np.ndarray, low: float, high: float) -> None:
    _require(bool(((arr >= low) & (arr <= high)).all()), f"{name}: values outside [{low}, {high}]")


def validate_ssfl_proposal_arrays(
    arrays: dict[str, np.ndarray], num_open: int, num_classes: int
) -> None:
    """Paper Algorithm 1 proposal: exactly one hard-label or soft-label payload.

    Confidence scores are strictly client-local discriminator/audit data and must not cross the
    wire in the canonical protocol.
    """
    _require("confidences" not in arrays, "confidences are client-local and forbidden on wire")
    pseudo_labels = arrays.get("pseudo_labels")
    soft_probs = arrays.get("soft_probs")
    _require(
        (pseudo_labels is None) != (soft_probs is None),
        "exactly one of pseudo_labels/soft_probs required",
    )

    if pseudo_labels is not None:
        _check_shape("pseudo_labels", pseudo_labels, (num_open,))
        _check_dtype_kind("pseudo_labels", pseudo_labels, "i")
        _require(
            bool(((pseudo_labels >= -1) & (pseudo_labels < num_classes)).all()),
            "pseudo_labels: value outside [-1, num_classes)",
        )

    if soft_probs is not None:
        _check_shape("soft_probs", soft_probs, (num_open, num_classes))
        _check_dtype_kind("soft_probs", soft_probs, "f")
        _check_finite("soft_probs", soft_probs)
        _check_range("soft_probs", soft_probs, -_PROB_TOL, 1 + _PROB_TOL)
        row_sums = soft_probs.sum(axis=1)
        # a legitimate row is either the "unfamiliar" all-zero sentinel or a softmax (~sums to 1).
        _require(
            bool(((row_sums < _PROB_TOL) | (np.abs(row_sums - 1) < _PROB_TOL)).all()),
            "soft_probs: rows must sum to ~1 or be all-zero",
        )


def validate_fd_arrays(arrays: dict[str, np.ndarray], num_classes: int) -> None:
    _require(
        "class_probs" in arrays and "class_present" in arrays, "missing class_probs/class_present"
    )
    class_probs = arrays["class_probs"]
    class_present = arrays["class_present"]
    _check_shape("class_probs", class_probs, (num_classes, num_classes))
    _check_dtype_kind("class_probs", class_probs, "f")
    _check_finite("class_probs", class_probs)
    _check_range("class_probs", class_probs, -_PROB_TOL, 1 + _PROB_TOL)

    _check_shape("class_present", class_present, (num_classes,))
    _check_dtype_kind("class_present", class_present, "i")
    _require(
        bool(((class_present == 0) | (class_present == 1)).all()),
        "class_present: values must be 0 or 1",
    )

    present_rows = class_probs[class_present.astype(bool)]
    if len(present_rows):
        row_sums = present_rows.sum(axis=1)
        _require(
            bool((np.abs(row_sums - 1) < _PROB_TOL).all()),
            "class_probs: present-class rows must sum to ~1",
        )


def validate_dsfl_arrays(arrays: dict[str, np.ndarray], num_open: int, num_classes: int) -> None:
    _require("probs" in arrays, "missing 'probs'")
    probs = arrays["probs"]
    _check_shape("probs", probs, (num_open, num_classes))
    _check_dtype_kind("probs", probs, "f")
    _check_finite("probs", probs)
    _check_range("probs", probs, -_PROB_TOL, 1 + _PROB_TOL)
    row_sums = probs.sum(axis=1)
    _require(bool((np.abs(row_sums - 1) < _PROB_TOL).all()), "probs: rows must sum to ~1")


if __name__ == "__main__":
    good = {"pseudo_labels": np.array([2, -1], dtype=np.int8)}
    validate_ssfl_proposal_arrays(good, num_open=2, num_classes=11)  # must not raise
    try:
        validate_ssfl_proposal_arrays({}, num_open=2, num_classes=11)
        raise SystemExit("expected ProtocolError for wrong shape")
    except ProtocolError:
        pass
    try:
        bad = {"pseudo_labels": np.array([2, 99], dtype=np.int8)}
        validate_ssfl_proposal_arrays(bad, num_open=2, num_classes=11)
        raise SystemExit("expected ProtocolError for out-of-range confidence")
    except ProtocolError:
        pass
    print("payload_limits.py self-check OK")
