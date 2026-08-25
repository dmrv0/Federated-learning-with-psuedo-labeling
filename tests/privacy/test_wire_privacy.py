"""Privacy-boundary tests beyond what tests/protocol/test_*.py already covers per-algorithm:
distillation-phase replies never carry model arrays, runtime data loaders are fully independent of
the sealed source-row audit trail, and each client's persisted model is separate from every other
client's (SSFL_IMPLEMENTATION_PLAN.md M9 "Protocol/security" test bullets).
"""

from __future__ import annotations

import dataclasses

import torch
from flwr.common import Context, RecordDict

from ssfl.client_app import _get_or_init_classifier, _save_model
from ssfl.config import Backbone, ExperimentConfig
from ssfl.data.datasets import load_client_assignments, load_open_data, load_test_data
from ssfl.training import TrainResult


def test_distillation_reply_never_carries_model_arrays() -> None:
    """SSFL phase-B, FD phase-B, and DS-FL phase-B client steps (client_distillation_step /
    fd.client_distillation_step / dsfl.distill_step) all return this one type. If it ever grows a
    model-shaped field, every distillation reply built from it would start leaking parameters.
    Telemetry metadata is allowed, but model/tensor fields are not."""
    field_names = {f.name for f in dataclasses.fields(TrainResult)}
    assert field_names == {
        "epoch_losses",
        "epoch_metrics",
        "total_examples",
        "total_batches",
        "duration_seconds",
    }
    instance = TrainResult(epoch_losses=[1.0, 0.5])
    for value in dataclasses.asdict(instance).values():
        assert not isinstance(value, torch.nn.Module)
        assert not isinstance(value, torch.Tensor)


def test_runtime_loaders_never_touch_sealed_audit(prepared_data_root) -> None:
    """audit/source_rows.parquet carries true open labels + raw source-row provenance and must
    never be read by anything on the training path. Proven behaviorally: move the audit directory
    away and confirm every runtime loader still works -- if a loader secretly depended on it, this
    would fail with FileNotFoundError instead of silently passing."""
    audit_dir = prepared_data_root / "audit"
    assert audit_dir.exists()
    moved = prepared_data_root / "audit_moved_away"
    audit_dir.rename(moved)
    try:
        open_ds = load_open_data(prepared_data_root, Backbone.cnn)
        assert open_ds.y is None
        test_ds = load_test_data(prepared_data_root, Backbone.cnn)
        assert len(test_ds) > 0
        assignments = load_client_assignments(prepared_data_root, scenario=1)
        assert len(assignments) == 27
    finally:
        moved.rename(audit_dir)


def test_client_model_persists_and_differs_across_clients() -> None:
    """context.state round-trips a client's classifier weights across calls (persistence), and two
    different clients' context.state never share weights after one of them trains (separation) --
    the two properties client_app.py's docstring claims for SSFL/FD/DS-FL local models."""
    exp_config = ExperimentConfig(backbone=Backbone.cnn, seed=4242)
    ctx_a = Context(run_id=0, node_id=1, node_config={}, state=RecordDict(), run_config={})
    ctx_b = Context(run_id=0, node_id=2, node_config={}, state=RecordDict(), run_config={})

    clf_a1 = _get_or_init_classifier(ctx_a, exp_config)
    clf_b1 = _get_or_init_classifier(ctx_b, exp_config)
    # Round-0 init is deterministic from the shared seed alone (never sent over the wire): both
    # clients must start from bit-identical weights.
    for pa, pb in zip(clf_a1.parameters(), clf_b1.parameters(), strict=True):
        assert torch.equal(pa, pb)

    with torch.no_grad():
        for p in clf_a1.parameters():
            p.add_(1.0)
    _save_model(ctx_a, "classifier", clf_a1)
    _save_model(ctx_b, "classifier", clf_b1)

    clf_a2 = _get_or_init_classifier(ctx_a, exp_config)
    clf_b2 = _get_or_init_classifier(ctx_b, exp_config)

    for pa1, pa2 in zip(clf_a1.parameters(), clf_a2.parameters(), strict=True):
        assert torch.equal(pa1, pa2)  # persisted, not reinitialized

    differs = any(
        not torch.equal(pa, pb)
        for pa, pb in zip(clf_a2.parameters(), clf_b2.parameters(), strict=True)
    )
    assert differs  # client A's trained weights never leaked into client B's state
