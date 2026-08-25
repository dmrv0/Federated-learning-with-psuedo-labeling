"""Versioned protocol state machine shared by SSFL/FL/FD/DS-FL (SSFL_IMPLEMENTATION_PLAN.md M4).

SSFL visits every state each round (two Message-API exchanges: proposal, distillation). FL/FD/
DS-FL skip ``LOCAL_DISTILLED`` on rounds where broadcast is immediately followed by evaluation
(FL: the new global weights ARE the model, no separate distillation step) -- the transition table
allows both paths rather than modelling four separate machines for one shared vocabulary.
"""

from __future__ import annotations

from enum import Enum

from ssfl.protocols.message import ProtocolError


class ProtocolState(str, Enum):
    initialized = "INITIALIZED"
    local_supervised = "LOCAL_SUPERVISED"
    proposal_uploaded = "PROPOSAL_UPLOADED"
    server_aggregated = "SERVER_AGGREGATED"
    global_target_broadcast = "GLOBAL_TARGET_BROADCAST"
    local_distilled = "LOCAL_DISTILLED"
    evaluated = "EVALUATED"
    checkpointed = "CHECKPOINTED"


_TRANSITIONS: dict[ProtocolState, frozenset[ProtocolState]] = {
    ProtocolState.initialized: frozenset({ProtocolState.local_supervised}),
    ProtocolState.local_supervised: frozenset({ProtocolState.proposal_uploaded}),
    ProtocolState.proposal_uploaded: frozenset({ProtocolState.server_aggregated}),
    ProtocolState.server_aggregated: frozenset({ProtocolState.global_target_broadcast}),
    ProtocolState.global_target_broadcast: frozenset({ProtocolState.local_distilled, ProtocolState.evaluated}),
    ProtocolState.local_distilled: frozenset({ProtocolState.evaluated}),
    ProtocolState.evaluated: frozenset({ProtocolState.checkpointed, ProtocolState.local_supervised}),
    ProtocolState.checkpointed: frozenset({ProtocolState.local_supervised}),
}


def validate_transition(current: ProtocolState, next_state: ProtocolState) -> None:
    if next_state not in _TRANSITIONS.get(current, frozenset()):
        raise ProtocolError(f"illegal protocol transition {current.value} -> {next_state.value}")
