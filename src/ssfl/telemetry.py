"""Durable, privacy-safe experiment telemetry.

Every event is one flushed JSON line.  Client processes write separate files, avoiding
cross-process interleaving, while the server writes the authoritative run/round stream.  Only
statistics, hashes, shapes, and timings are recorded: private feature values, gradients, model
weights, and prediction payload contents are deliberately excluded.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

EventCallback = Callable[[str, dict[str, Any]], None]


def filter_batch_events(
    callback: EventCallback | None, *, log_every_batch: bool
) -> EventCallback | None:
    """Apply the configured batch-event policy without suppressing higher-level telemetry."""
    if callback is None or log_every_batch:
        return callback

    def _filtered(event: str, fields: dict[str, Any]) -> None:
        if not event.endswith("_batch"):
            callback(event, fields)

    return _filtered


def gpu_snapshot() -> dict[str, Any]:
    """Process-local CUDA allocator state; empty on non-CUDA hosts."""
    if not torch.cuda.is_available():
        return {}
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    return {
        "cuda_device": device,
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_memory_free_bytes": int(free),
        "gpu_memory_total_bytes": int(total),
        "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "gpu_memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "gpu_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "gpu_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def nvidia_smi_snapshot() -> list[dict[str, Any]]:
    """Whole-device utilization/thermal/power sample; best effort and never fatal."""
    fields = [
        "index",
        "uuid",
        "name",
        "driver_version",
        "temperature.gpu",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "power.draw",
        "clocks.current.graphics",
        "clocks.current.memory",
    ]
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == len(fields):
            rows.append(dict(zip(fields, values, strict=True)))
    return rows


class JsonlEventWriter:
    """Append exactly one flushed JSON object per call."""

    def __init__(self, path: Path, **bound: Any) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.bound = bound
        self._lock = threading.Lock()

    def emit(self, event: str, **fields: Any) -> None:
        payload = {
            "event": event,
            "time_unix_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            **self.bound,
            **fields,
        }
        line = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()

    def callback(self, **bound: Any) -> EventCallback:
        def _callback(event: str, fields: dict[str, Any]) -> None:
            self.emit(event, **bound, **fields)

        return _callback


class SystemMonitor:
    """Background NVIDIA/system sampler owned by the ServerApp process."""

    def __init__(self, writer: JsonlEventWriter, interval_seconds: float) -> None:
        self.writer = writer
        self.interval_seconds = max(float(interval_seconds), 0.2)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ssfl-system-monitor", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(5.0, self.interval_seconds * 2))

    def _run(self) -> None:
        while not self._stop.is_set():
            self.writer.emit(
                "system_sample",
                load_average=list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
                cuda_process=gpu_snapshot(),
                nvidia_smi=nvidia_smi_snapshot(),
            )
            self._stop.wait(self.interval_seconds)
