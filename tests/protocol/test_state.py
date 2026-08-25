import pytest

from ssfl.protocols.message import ProtocolError
from ssfl.protocols.state import ProtocolState, validate_transition


def test_valid_full_ssfl_cycle() -> None:
    cycle = [
        ProtocolState.initialized,
        ProtocolState.local_supervised,
        ProtocolState.proposal_uploaded,
        ProtocolState.server_aggregated,
        ProtocolState.global_target_broadcast,
        ProtocolState.local_distilled,
        ProtocolState.evaluated,
        ProtocolState.checkpointed,
        ProtocolState.local_supervised,
    ]
    for a, b in zip(cycle, cycle[1:]):
        validate_transition(a, b)  # must not raise


def test_non_ssfl_can_skip_local_distilled() -> None:
    validate_transition(ProtocolState.global_target_broadcast, ProtocolState.evaluated)


def test_evaluated_can_loop_without_checkpointing() -> None:
    validate_transition(ProtocolState.evaluated, ProtocolState.local_supervised)


def test_illegal_transition_rejected() -> None:
    with pytest.raises(ProtocolError):
        validate_transition(ProtocolState.initialized, ProtocolState.evaluated)


def test_cannot_go_backwards() -> None:
    with pytest.raises(ProtocolError):
        validate_transition(ProtocolState.evaluated, ProtocolState.proposal_uploaded)


def test_checkpointed_can_only_resume_at_local_supervised() -> None:
    with pytest.raises(ProtocolError):
        validate_transition(ProtocolState.checkpointed, ProtocolState.evaluated)
