"""Device resolution for CPU / CUDA / Apple MPS.

Per REPRODUCIBILITY.md assumption #12: MPS is not treated as a deterministic backend. Resolving
``auto`` never silently picks MPS when the caller asked for deterministic execution — it logs why
and falls back to CPU instead, so the *paper* profile's reproducibility guarantee never depends on
an unverified accelerator.
"""

from __future__ import annotations

import logging

import torch

from ssfl.config import DeviceKind

logger = logging.getLogger("ssfl.device")


def _cpu_device() -> torch.device:
    # Simulation runs many client processes (Ray actors) concurrently, each defaulting to a
    # full-core OMP thread pool; that oversubscribes the machine and the fork/join barrier
    # overhead dwarfs actual compute for this model's tiny tensors. One thread per process is
    # faster in aggregate than N threads x N concurrent processes.
    torch.set_num_threads(1)
    return torch.device("cpu")


def resolve_device(requested: DeviceKind, deterministic: bool) -> torch.device:
    if requested == DeviceKind.cpu:
        return _cpu_device()

    if requested == DeviceKind.cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but torch.cuda.is_available() is False")
        return torch.device("cuda")

    if requested == DeviceKind.mps:
        if not torch.backends.mps.is_available():
            raise RuntimeError("device=mps requested but torch.backends.mps.is_available() is False")
        if deterministic:
            logger.warning(
                "device=mps explicitly requested under deterministic=True; MPS float32 kernels are "
                "not verified bit-reproducible across runs (see REPRODUCIBILITY.md #12). Proceeding "
                "at the caller's request, but this run cannot be used for the CPU-smoke determinism "
                "acceptance gate."
            )
        return torch.device("mps")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if deterministic:
        if torch.backends.mps.is_available():
            logger.info(
                "device=auto with deterministic=True: MPS is available but not used for "
                "determinism reasons (see REPRODUCIBILITY.md #12); falling back to cpu."
            )
        return _cpu_device()
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return _cpu_device()
