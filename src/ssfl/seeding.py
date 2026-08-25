"""Deterministic seeding for Python, NumPy, PyTorch (CPU/CUDA), and DataLoader workers.

Every entrypoint that trains or partitions data must call :func:`seed_everything` first with the
run's resolved seed before touching any randomness-consuming API, so that partitioning, model
initialization, and local training are all reproducible from the same seed.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger("ssfl.seeding")


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dataloader_worker_init_fn(worker_id: int) -> None:
    """Derive a distinct, deterministic seed per DataLoader worker from the base torch seed."""
    base_seed = torch.initial_seed() % (2**32)
    worker_seed = (base_seed + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """A seeded ``torch.Generator`` for DataLoader ``generator=`` (keeps worker shuffling
    reproducible independent of global RNG state mutated by other code between calls)."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def configure_determinism(deterministic: bool) -> list[str]:
    """Enable deterministic PyTorch kernels where requested; return warnings for any op that falls
    back to a nondeterministic implementation (``warn_only=True`` so we can log rather than crash
    on ops without a deterministic kernel, then surface the fallback explicitly).
    """
    warnings: list[str] = []
    if not deterministic:
        torch.use_deterministic_algorithms(False)
        return warnings

    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import warnings as _warnings_module

    with _warnings_module.catch_warnings(record=True) as caught:
        _warnings_module.simplefilter("always")
        torch.use_deterministic_algorithms(True, warn_only=True)
        for w in caught:
            msg = str(w.message)
            warnings.append(msg)
            logger.warning("nondeterministic fallback: %s", msg)
    return warnings
