"""Flower-independent message envelope + validation shared by all four protocols.

Every algorithm's client->server / server->client exchange is wrapped in an :class:`Envelope`
before touching payload content. Payload shape/dtype checks are algorithm-specific and live next
to the aggregation logic that consumes them (``protocols/ssfl.py`` etc.); this module only
enforces the envelope-level safeguards common to all four (M4 "Protocol safeguards").

Flower's actual ``Message``/``RecordDict`` wiring happens in M5 -- this module has no ``flwr``
import, satisfying the M4 gate ("independent protocol tests pass without starting Flower").
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from ssfl.config import Algorithm

PROTOCOL_VERSION = 1


class ProtocolError(Exception):
    """Raised when a message fails envelope validation; callers catch this per-message and
    record it as a rejection rather than crashing the whole round."""


@dataclass(frozen=True)
class Envelope:
    algorithm: Algorithm
    scenario: int
    round: int
    phase: str
    sender_id: str
    dataset_manifest_hash: str
    payload: dict[str, Any] = field(default_factory=dict)
    protocol_version: int = PROTOCOL_VERSION

    @property
    def message_id(self) -> str:
        """Deterministic over (algorithm, scenario, round, phase, sender) -- NOT payload content,
        so a retried/redelivered copy of the same logical message shares one id (idempotent
        dedup), while two genuinely different submissions for the same slot collide and are
        treated as a duplicate rather than both being aggregated."""
        key = f"{self.algorithm.value}|{self.scenario}|{self.round}|{self.phase}|{self.sender_id}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ExpectedContext:
    algorithm: Algorithm
    scenario: int
    round: int
    phase: str
    dataset_manifest_hash: str
    valid_senders: frozenset[str]


def validate_envelope(msg: Envelope, expected: ExpectedContext, seen_message_ids: set[str]) -> None:
    """Raises :class:`ProtocolError` on the first failing check. Order matters only for error
    message clarity; callers that need to record rejection *reasons* should catch and inspect
    ``str(exc)``."""
    if msg.protocol_version != PROTOCOL_VERSION:
        raise ProtocolError(f"protocol_version {msg.protocol_version} != {PROTOCOL_VERSION}")
    if msg.algorithm != expected.algorithm:
        raise ProtocolError(f"wrong algorithm: got {msg.algorithm.value}, expected {expected.algorithm.value}")
    if msg.scenario != expected.scenario:
        raise ProtocolError(f"wrong scenario: got {msg.scenario}, expected {expected.scenario}")
    if msg.dataset_manifest_hash != expected.dataset_manifest_hash:
        raise ProtocolError("dataset_manifest_hash mismatch")
    if msg.sender_id not in expected.valid_senders:
        raise ProtocolError(f"unknown sender {msg.sender_id!r}")
    if msg.phase != expected.phase:
        raise ProtocolError(f"wrong phase: got {msg.phase!r}, expected {expected.phase!r}")
    if msg.round < expected.round:
        raise ProtocolError(f"stale round {msg.round} < expected {expected.round}")
    if msg.round > expected.round:
        raise ProtocolError(f"future round {msg.round} > expected {expected.round}")
    if msg.message_id in seen_message_ids:
        raise ProtocolError(f"duplicate message {msg.message_id} from {msg.sender_id!r}")
