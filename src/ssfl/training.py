"""Shared supervised + distillation training loops, evaluation, and batched open-data
prediction. Independent of Flower -- protocols (M4) call these directly against local models.
"""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ssfl.seeding import dataloader_worker_init_fn, make_generator
from ssfl.telemetry import EventCallback, gpu_snapshot


@dataclass
class TrainResult:
    epoch_losses: list[float] = field(default_factory=list)
    epoch_metrics: list[dict[str, Any]] = field(default_factory=list)
    total_examples: int = 0
    total_batches: int = 0
    duration_seconds: float = 0.0

    @property
    def final_loss(self) -> float:
        return self.epoch_losses[-1] if self.epoch_losses else float("nan")


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    """Deterministic DataLoader: a seeded generator drives shuffling and worker seeding
    independent of whatever else has touched global RNG state."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=make_generator(seed) if shuffle else None,
        worker_init_fn=dataloader_worker_init_fn if shuffle else None,
    )


def build_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    """Fresh Adam every call -- SSFL_IMPLEMENTATION_PLAN.md M3: 'fresh optimizer per Flower task
    while model weights persist.'"""
    return torch.optim.Adam(model.parameters(), lr=lr)


def _amp_context(device: torch.device, enabled: bool):
    # ponytail: mixed precision is a CUDA-only perf profile, off by default (paper mode never
    # sets it); on CPU/MPS this silently no-ops rather than erroring on unsupported autocast.
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda")
    return contextlib.nullcontext()


def _run_epochs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    loss_fn,
    grad_clip_norm: float | None,
    use_amp: bool,
    event_callback: EventCallback | None = None,
    stage: str = "train",
) -> TrainResult:
    model.to(device)
    model.train()
    optimizer = build_optimizer(model, lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    result = TrainResult()
    train_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if event_callback:
        event_callback(
            "training_start",
            {
                "stage": stage,
                "epochs": epochs,
                "learning_rate": lr,
                "num_batches": len(loader) if hasattr(loader, "__len__") else None,
                "mixed_precision": use_amp,
                "grad_clip_norm": grad_clip_norm,
                "device": str(device),
                **gpu_snapshot(),
            },
        )
    global_batch = 0
    for epoch_index in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        total_loss, total_count, correct = 0.0, 0, 0
        has_hard_targets = False
        batch_losses: list[float] = []
        for batch_index, (x, target) in enumerate(loader, start=1):
            batch_started = time.perf_counter()
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            with _amp_context(device, use_amp):
                logits = model(x)
                loss = loss_fn(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_sq = sum(
                float(parameter.grad.detach().float().pow(2).sum().item())
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            grad_norm = math.sqrt(grad_sq)
            if grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            batch_loss = float(loss.item())
            batch_size_actual = int(x.shape[0])
            total_loss += batch_loss * batch_size_actual
            total_count += batch_size_actual
            batch_losses.append(batch_loss)
            if target.ndim == 1 and not target.dtype.is_floating_point:
                has_hard_targets = True
                correct += int((logits.argmax(dim=-1) == target).sum().item())
            parameter_sq = sum(
                float(parameter.detach().float().pow(2).sum().item())
                for parameter in model.parameters()
            )
            global_batch += 1
            result.total_batches += 1
            result.total_examples += batch_size_actual
            if event_callback:
                event_callback(
                    "training_batch",
                    {
                        "stage": stage,
                        "epoch": epoch_index,
                        "batch": batch_index,
                        "global_batch": global_batch,
                        "batch_size": batch_size_actual,
                        "loss": batch_loss,
                        "gradient_l2_norm": grad_norm,
                        "parameter_l2_norm": math.sqrt(parameter_sq),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "scaler_scale": float(scaler.get_scale()),
                        "duration_seconds": time.perf_counter() - batch_started,
                        **gpu_snapshot(),
                    },
                )
        epoch_loss = total_loss / total_count
        epoch_record = {
            "stage": stage,
            "epoch": epoch_index,
            "loss": epoch_loss,
            "loss_min_batch": min(batch_losses),
            "loss_max_batch": max(batch_losses),
            "loss_std_batch": float(np.std(batch_losses)),
            "accuracy": correct / total_count if has_hard_targets else None,
            "examples": total_count,
            "batches": len(batch_losses),
            "duration_seconds": time.perf_counter() - epoch_started,
            **gpu_snapshot(),
        }
        result.epoch_losses.append(epoch_loss)
        result.epoch_metrics.append(epoch_record)
        if event_callback:
            event_callback("training_epoch", epoch_record)
    result.duration_seconds = time.perf_counter() - train_started
    if event_callback:
        event_callback(
            "training_end",
            {
                "stage": stage,
                "final_loss": result.final_loss,
                "duration_seconds": result.duration_seconds,
                "total_examples": result.total_examples,
                "total_batches": result.total_batches,
                **gpu_snapshot(),
            },
        )
    return result


def train_supervised(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    grad_clip_norm: float | None = None,
    use_amp: bool = False,
    event_callback: EventCallback | None = None,
    stage: str = "supervised",
) -> TrainResult:
    """Hard-label cross-entropy training (loader yields ``(x, class_index)``)."""
    return _run_epochs(
        model,
        loader,
        device,
        epochs,
        lr,
        nn.CrossEntropyLoss(),
        grad_clip_norm,
        use_amp,
        event_callback,
        stage,
    )


def teacher_distribution_loss(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    return -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def train_distillation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    grad_clip_norm: float | None = None,
    use_amp: bool = False,
    event_callback: EventCallback | None = None,
    stage: str = "distillation",
) -> TrainResult:
    """Soft-target teacher-distribution loss (loader yields ``(x, prob_distribution)``) -- used
    for FD/DS-FL distillation and SSFL's soft-label ablation."""
    return _run_epochs(
        model,
        loader,
        device,
        epochs,
        lr,
        teacher_distribution_loss,
        grad_clip_norm,
        use_amp,
        event_callback,
        stage,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    event_callback: EventCallback | None = None,
    stage: str = "evaluation",
) -> dict:
    """Loss/accuracy plus ``y_true``/``y_pred`` (loader order) so callers can compute richer
    metrics (macro P/R/F1, confusion matrix -- see ``metrics.py``) without a second pass."""
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []
    started = time.perf_counter()
    for batch_index, (x, y) in enumerate(loader, start=1):
        batch_started = time.perf_counter()
        x, y = x.to(device), y.to(device)
        logits = model(x)
        batch_loss_sum = float(criterion(logits, y).item())
        total_loss += batch_loss_sum
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += x.shape[0]
        y_true.append(y.cpu().numpy())
        y_pred.append(preds.cpu().numpy())
        if event_callback:
            event_callback(
                "evaluation_batch",
                {
                    "stage": stage,
                    "batch": batch_index,
                    "batch_size": int(x.shape[0]),
                    "loss_sum": batch_loss_sum,
                    "correct": int((preds == y).sum().item()),
                    "duration_seconds": time.perf_counter() - batch_started,
                    **gpu_snapshot(),
                },
            )
    output = {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "y_true": np.concatenate(y_true) if y_true else np.array([], dtype=np.int64),
        "y_pred": np.concatenate(y_pred) if y_pred else np.array([], dtype=np.int64),
    }
    if event_callback:
        event_callback(
            "evaluation_end",
            {
                "stage": stage,
                "loss": output["loss"],
                "accuracy": output["accuracy"],
                "examples": total,
                "duration_seconds": time.perf_counter() - started,
                **gpu_snapshot(),
            },
        )
    return output


@torch.no_grad()
def predict_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    event_callback: EventCallback | None = None,
    stage: str = "prediction",
) -> torch.Tensor:
    """Batched softmax probabilities over an unlabeled loader (open data), in loader order."""
    model.to(device)
    model.eval()
    chunks = []
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        batch_started = time.perf_counter()
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        probs = torch.softmax(model(x.to(device)), dim=-1).cpu()
        chunks.append(probs)
        if event_callback:
            event_callback(
                "prediction_batch",
                {
                    "stage": stage,
                    "batch": batch_index,
                    "batch_size": int(probs.shape[0]),
                    "confidence_mean": float(probs.max(dim=1).values.mean().item()),
                    "duration_seconds": time.perf_counter() - batch_started,
                    **gpu_snapshot(),
                },
            )
    output = torch.cat(chunks, dim=0)
    if event_callback:
        event_callback(
            "prediction_end",
            {
                "stage": stage,
                "examples": int(output.shape[0]),
                "duration_seconds": time.perf_counter() - started,
                **gpu_snapshot(),
            },
        )
    return output
