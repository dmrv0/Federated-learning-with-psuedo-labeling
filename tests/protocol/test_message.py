import pytest

from ssfl.config import Algorithm
from ssfl.protocols.message import Envelope, ExpectedContext, ProtocolError, validate_envelope


def _envelope(**overrides):
    base = dict(
        algorithm=Algorithm.ssfl,
        scenario=1,
        round=1,
        phase="proposal",
        sender_id="c0",
        dataset_manifest_hash="abc123",
    )
    base.update(overrides)
    return Envelope(**base)


def _expected(**overrides):
    base = dict(
        algorithm=Algorithm.ssfl,
        scenario=1,
        round=1,
        phase="proposal",
        dataset_manifest_hash="abc123",
        valid_senders=frozenset({"c0", "c1"}),
    )
    base.update(overrides)
    return ExpectedContext(**base)


def test_valid_envelope_passes() -> None:
    validate_envelope(_envelope(), _expected(), seen_message_ids=set())


def test_rejects_wrong_algorithm() -> None:
    with pytest.raises(ProtocolError):
        validate_envelope(_envelope(), _expected(algorithm=Algorithm.fl), set())


def test_rejects_wrong_scenario() -> None:
    with pytest.raises(ProtocolError):
        validate_envelope(_envelope(), _expected(scenario=2), set())


def test_rejects_dataset_hash_mismatch() -> None:
    with pytest.raises(ProtocolError):
        validate_envelope(_envelope(), _expected(dataset_manifest_hash="different"), set())


def test_rejects_unknown_sender() -> None:
    with pytest.raises(ProtocolError):
        validate_envelope(_envelope(sender_id="unknown"), _expected(), set())


def test_rejects_wrong_phase() -> None:
    with pytest.raises(ProtocolError):
        validate_envelope(_envelope(), _expected(phase="distillation"), set())


def test_rejects_stale_round() -> None:
    with pytest.raises(ProtocolError):
        validate_envelope(_envelope(round=1), _expected(round=2), set())


def test_rejects_future_round() -> None:
    with pytest.raises(ProtocolError):
        validate_envelope(_envelope(round=3), _expected(round=2), set())


def test_rejects_duplicate_message_id() -> None:
    envelope = _envelope()
    seen = {envelope.message_id}
    with pytest.raises(ProtocolError):
        validate_envelope(envelope, _expected(), seen)


def test_rejects_wrong_protocol_version() -> None:
    bad = Envelope(**{**_envelope().__dict__, "protocol_version": 999})
    with pytest.raises(ProtocolError):
        validate_envelope(bad, _expected(), set())


def test_message_id_stable_across_payload_content() -> None:
    e1 = _envelope(payload={"a": 1})
    e2 = _envelope(payload={"a": 2})
    assert e1.message_id == e2.message_id  # a retry with the same content shares one id


def test_message_id_differs_by_sender() -> None:
    assert _envelope(sender_id="c0").message_id != _envelope(sender_id="c1").message_id


def test_message_id_differs_by_round() -> None:
    assert _envelope(round=1).message_id != _envelope(round=2).message_id
