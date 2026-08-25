"""Flower ``ServerApp`` entrypoint: builds the per-algorithm ``Strategy`` and drives
``strategy.start(...)``, which runs the full round loop (train exchange -> aggregate -> evaluate
exchange -> aggregate -> optional centralized eval) for all four algorithms uniformly.

FL reuses the built-in ``FedAvg`` unmodified (a stock aggregation pattern with no SSFL-specific
safety property to enforce beyond what FedAvg's own reply-consistency checks already guarantee).
SSFL/FD/DS-FL use the custom strategies in ``strategies/`` since none of them ever transmits model
parameters, which ``FedAvg`` assumes.
"""

from __future__ import annotations

import json

from flwr.common import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from ssfl.comms import CommsTrackingStrategy
from ssfl.config import Algorithm, experiment_config_from_run_config
from ssfl.data.datasets import load_open_data, load_test_data
from ssfl.device import resolve_device
from ssfl.logging_utils import bind, configure_logging, log_event
from ssfl.metrics import MetricsLedger, compute_classification_metrics
from ssfl.models import NUM_CLASSES, build_classifier
from ssfl.protocols import dsfl as dsfl_protocol
from ssfl.protocols import fl as fl_protocol
from ssfl.protocols.ssfl import AggregationResult, server_distillation_step
from ssfl.records import array_record_from_numpy, numpy_from_array_record
from ssfl.run_context import RunContext, prune_superseded_checkpoints
from ssfl.seeding import configure_determinism, seed_everything
from ssfl.strategies.dsfl import DSFLStrategy
from ssfl.strategies.fd import FDStrategy
from ssfl.strategies.ssfl import SSFLStrategy
from ssfl.telemetry import JsonlEventWriter, SystemMonitor, filter_batch_events, gpu_snapshot

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    requested_config = experiment_config_from_run_config(context.run_config)
    if requested_config.resume_from is not None:
        run_context = RunContext.resume(requested_config.resume_from)
        exp_config = run_context.config
        resume_checkpoint = run_context.load_latest_server_checkpoint()
    else:
        exp_config = requested_config
        manifest_path = exp_config.data_path / "dataset_manifest.json"
        run_context = RunContext.create(exp_config, dataset_manifest_path=manifest_path)
        resume_checkpoint = None
    device = resolve_device(exp_config.device, exp_config.deterministic)
    configure_determinism(exp_config.deterministic)

    manifest_path = exp_config.data_path / "dataset_manifest.json"
    manifest_hash = None
    if manifest_path.exists():
        manifest_hash = json.loads(manifest_path.read_text()).get("manifest_hash")
    metrics_ledger = MetricsLedger(
        run_dir=run_context.run_dir,
        load_existing=resume_checkpoint is not None,
        completed_through=int(resume_checkpoint["round"]) if resume_checkpoint else None,
    )

    telemetry = JsonlEventWriter(
        run_context.telemetry_dir / "server.jsonl",
        run_id=run_context.run_id,
        attempt_id=run_context.attempt_id,
        algorithm=exp_config.algorithm.value,
        scenario=exp_config.scenario.value,
        role="server",
    )
    server_callback = filter_batch_events(
        telemetry.callback(phase="server"),
        log_every_batch=exp_config.log_every_batch,
    )
    monitor = SystemMonitor(telemetry, exp_config.system_monitor_interval_seconds)
    monitor.start()

    events_stream = open(run_context.events_path, "a", buffering=1)
    events_logger = bind(
        configure_logging(stream=events_stream),
        run_id=run_context.run_id,
        algorithm=exp_config.algorithm.value,
        scenario=exp_config.scenario.value,
        attempt_id=run_context.attempt_id,
    )
    log_event(
        events_logger,
        "run_start",
        dataset_hash=manifest_hash,
        device=str(device),
        resume_round=resume_checkpoint["round"] if resume_checkpoint else 0,
    )
    telemetry.emit(
        "run_start",
        dataset_hash=manifest_hash,
        resolved_config=exp_config.model_dump(mode="json"),
        device=str(device),
        resume_round=resume_checkpoint["round"] if resume_checkpoint else 0,
        **gpu_snapshot(),
    )

    shared_config = {
        "run-id": run_context.run_id,
        "run-dir": str(run_context.run_dir.resolve()),
        "attempt-id": run_context.attempt_id,
        "attempt-dir": str(run_context.attempt_dir.resolve()),
    }
    train_config = ConfigRecord(shared_config.copy())
    evaluate_config = ConfigRecord(shared_config.copy())
    server_classifier = None

    if exp_config.algorithm is Algorithm.fl:
        seed_everything(exp_config.seed)
        initial_arrays = ArrayRecord(
            torch_state_dict=build_classifier(exp_config.backbone).state_dict()
        )
        strategy = FedAvg(
            fraction_train=1.0,
            fraction_evaluate=1.0,
            min_train_nodes=exp_config.num_clients(),
            min_evaluate_nodes=exp_config.num_clients(),
            min_available_nodes=exp_config.num_clients(),
        )
        test_dataset = load_test_data(exp_config.data_path, exp_config.backbone)
        server_classifier = build_classifier(exp_config.backbone)

        def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord | None:
            eval_metrics = fl_protocol.server_evaluate(
                server_classifier,
                arrays.to_torch_state_dict(),
                test_dataset,
                device,
                batch_size=exp_config.effective_batch_size,
                seed=exp_config.seed,
                event_callback=server_callback,
            )
            classification = compute_classification_metrics(
                eval_metrics["y_true"], eval_metrics["y_pred"], NUM_CLASSES
            )
            telemetry.emit(
                "classification_metrics",
                round=server_round,
                **classification.to_dict(include_arrays=True),
            )
            metrics_ledger.record(
                algorithm=exp_config.algorithm.value,
                scenario=exp_config.scenario.value,
                round=server_round,
                loss=eval_metrics["loss"],
                metrics=classification,
            )
            return MetricRecord(
                {"loss": eval_metrics["loss"], "accuracy": eval_metrics["accuracy"]}
            )

    elif exp_config.algorithm is Algorithm.ssfl:
        if manifest_hash is None:
            raise RuntimeError(f"dataset_manifest.json missing or unhashed at {manifest_path}")
        open_dataset = load_open_data(exp_config.data_path, exp_config.backbone)
        test_dataset = load_test_data(exp_config.data_path, exp_config.backbone)
        strategy = SSFLStrategy(
            scenario=exp_config.scenario.value,
            dataset_manifest_hash=manifest_hash,
            num_open=len(open_dataset),
            num_clients=exp_config.num_clients(),
            voting_mode=exp_config.ssfl_voting_mode,
            audit_dir=run_context.attempt_dir / "aggregation_audit",
        )
        initial_arrays = ArrayRecord()
        seed_everything(exp_config.seed)
        server_classifier = build_classifier(exp_config.backbone)

        def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord | None:
            if server_round == 0:
                return None
            arrays_np = numpy_from_array_record(arrays)
            valid_mask = arrays_np["valid_mask"].astype(bool)
            aggregation = AggregationResult(
                global_labels=arrays_np["global_labels"],
                valid_mask=valid_mask,
                votes_per_class=None,
                participating_counts=None,
                tie_count=0,
                all_abstain_count=0,
                rejected=(),
            )
            train_result, eval_metrics = server_distillation_step(
                server_classifier,
                open_dataset,
                aggregation,
                test_dataset,
                device,
                epochs=exp_config.local_epochs,
                lr=exp_config.learning_rate,
                batch_size=exp_config.effective_batch_size,
                seed=exp_config.seed,
                event_callback=server_callback,
            )
            distillation_skipped = train_result.total_examples == 0
            train_loss = train_result.final_loss if not distillation_skipped else 0.0
            classification = compute_classification_metrics(
                eval_metrics["y_true"], eval_metrics["y_pred"], NUM_CLASSES
            )
            telemetry.emit(
                "classification_metrics",
                round=server_round,
                distillation_skipped=distillation_skipped,
                distillation_examples=train_result.total_examples,
                **classification.to_dict(include_arrays=True),
            )
            metrics_ledger.record(
                algorithm=exp_config.algorithm.value,
                scenario=exp_config.scenario.value,
                round=server_round,
                loss=eval_metrics["loss"],
                train_loss=train_loss,
                metrics=classification,
                valid_rate=float(valid_mask.mean()),
            )
            return MetricRecord(
                {
                    "loss": eval_metrics["loss"],
                    "accuracy": eval_metrics["accuracy"],
                    "train_loss": train_loss,
                    "distillation_skipped": int(distillation_skipped),
                    "distillation_examples": train_result.total_examples,
                }
            )

    elif exp_config.algorithm is Algorithm.fd:
        strategy = FDStrategy(num_clients=exp_config.num_clients())
        initial_arrays = ArrayRecord()
        evaluate_fn = None  # FD has no server-side global model; see strategies/fd.py docstring.

    elif exp_config.algorithm is Algorithm.dsfl:
        open_dataset = load_open_data(exp_config.data_path, exp_config.backbone)
        test_dataset = load_test_data(exp_config.data_path, exp_config.backbone)
        strategy = DSFLStrategy(
            temperature=exp_config.dsfl_temperature,
            num_clients=exp_config.num_clients(),
            num_open=len(open_dataset),
        )
        initial_arrays = ArrayRecord()
        seed_everything(exp_config.seed)
        server_classifier = build_classifier(exp_config.backbone)

        def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord | None:
            if server_round == 0:
                return None
            arrays_np = numpy_from_array_record(arrays)
            train_result = dsfl_protocol.distill_step(
                server_classifier,
                open_dataset,
                arrays_np["sharpened_targets"],
                device,
                epochs=exp_config.local_epochs,
                lr=exp_config.learning_rate,
                batch_size=exp_config.effective_batch_size,
                seed=exp_config.seed,
                event_callback=server_callback,
            )
            eval_metrics = dsfl_protocol.server_evaluate(
                server_classifier,
                test_dataset,
                device,
                batch_size=exp_config.effective_batch_size,
                seed=exp_config.seed,
                event_callback=server_callback,
            )
            classification = compute_classification_metrics(
                eval_metrics["y_true"], eval_metrics["y_pred"], NUM_CLASSES
            )
            telemetry.emit(
                "classification_metrics",
                round=server_round,
                **classification.to_dict(include_arrays=True),
            )
            metrics_ledger.record(
                algorithm=exp_config.algorithm.value,
                scenario=exp_config.scenario.value,
                round=server_round,
                loss=eval_metrics["loss"],
                train_loss=train_result.final_loss,
                metrics=classification,
            )
            return MetricRecord(
                {
                    "loss": eval_metrics["loss"],
                    "accuracy": eval_metrics["accuracy"],
                    "train_loss": train_result.final_loss,
                }
            )

    else:
        raise ValueError(f"unknown algorithm {exp_config.algorithm}")

    start_round = 1
    if resume_checkpoint is not None:
        initial_arrays = array_record_from_numpy(resume_checkpoint["arrays"])
        if server_classifier is not None and resume_checkpoint["server_model_state"] is not None:
            server_classifier.load_state_dict(resume_checkpoint["server_model_state"])
        start_round = int(resume_checkpoint["round"]) + 1
        telemetry.emit("checkpoint_restored", round=start_round - 1)

    if start_round > exp_config.num_server_rounds:
        telemetry.emit("run_already_complete", completed_round=start_round - 1)
        log_event(events_logger, "run_already_complete", completed_round=start_round - 1)
        monitor.stop()
        events_stream.close()
        return

    def on_round_end(server_round, arrays, round_result) -> None:
        if exp_config.algorithm is Algorithm.fd:
            values = round_result.evaluate_metrics_clientapp.get(server_round)
            if values is not None:
                metrics_ledger.record_precomputed(
                    algorithm=exp_config.algorithm.value,
                    scenario=exp_config.scenario.value,
                    round=server_round,
                    loss=float(values["loss"]),
                    accuracy=float(values["accuracy"]),
                    macro_precision=float(values["macro_precision"]),
                    macro_recall=float(values["macro_recall"]),
                    macro_f1=float(values["macro_f1"]),
                    selected_client=int(values["selected_client"]),
                )
        if not run_context.checkpoint_due(server_round):
            return
        model_state = (
            {key: value.detach().cpu() for key, value in server_classifier.state_dict().items()}
            if server_classifier is not None
            else None
        )
        checkpoint_path = run_context.save_server_checkpoint(
            server_round,
            numpy_from_array_record(arrays),
            model_state,
            extra={"attempt_id": run_context.attempt_id},
        )
        telemetry.emit("checkpoint_written", round=server_round, path=str(checkpoint_path))
        pruned_paths = prune_superseded_checkpoints(
            run_context.checkpoints_dir,
            current_round=server_round,
            pinned_rounds=exp_config.checkpoint_rounds,
        )
        if pruned_paths:
            telemetry.emit(
                "checkpoints_pruned",
                round=server_round,
                count=len(pruned_paths),
                paths=[str(path) for path in pruned_paths],
                retention="milestones_plus_latest",
            )

    tracked_strategy = CommsTrackingStrategy(
        strategy,
        exp_config.algorithm.value,
        exp_config.scenario.value,
        logger=events_logger,
        telemetry=telemetry,
        ledger_path=run_context.run_dir / "communication.parquet",
        attempt_id=run_context.attempt_id,
        load_existing=resume_checkpoint is not None,
        completed_through=start_round - 1 if resume_checkpoint is not None else None,
        start_round=start_round,
        round_end_callback=on_round_end,
    )
    try:
        result = tracked_strategy.start(
            grid,
            initial_arrays,
            num_rounds=exp_config.num_server_rounds,
            train_config=train_config,
            evaluate_config=evaluate_config,
            evaluate_fn=evaluate_fn,
        )
    finally:
        monitor.stop()
    tracked_strategy.ledger.write_parquet(run_context.run_dir / "communication.parquet")
    metrics_ledger.write(run_context.run_dir)

    final_metrics_by_round = (
        result.evaluate_metrics_clientapp
        if exp_config.algorithm is Algorithm.fd
        else result.evaluate_metrics_serverapp
    )
    final_round = max(final_metrics_by_round) if final_metrics_by_round else None
    run_context.write_summary(
        {
            "run_id": run_context.run_id,
            "algorithm": exp_config.algorithm.value,
            "scenario": exp_config.scenario.value,
            "num_rounds": exp_config.num_server_rounds,
            "final_round": final_round,
            "final_centralized_metrics": (
                dict(final_metrics_by_round[final_round].items())
                if final_round is not None
                else None
            ),
            "attempt_id": run_context.attempt_id,
            "resumed_from_round": start_round - 1,
        }
    )
    log_event(events_logger, "run_end", final_round=final_round)
    telemetry.emit("run_end", final_round=final_round, **gpu_snapshot())
    events_stream.close()
