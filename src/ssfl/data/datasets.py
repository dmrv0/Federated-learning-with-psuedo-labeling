"""Torch ``Dataset`` loaders over prepared SSFL artifacts (private/open/test/scenarios).

Reads only what ``ssfl.data.prepare_data`` wrote for training consumption: never touches the
sealed ``audit/source_rows.parquet`` (open rows carry no label field on disk -- see DATA_CARD.md
leakage-risk section; loading it here would defeat that boundary).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ssfl.config import Backbone
from ssfl.data.partition import ClientAssignment


def _select_features(flat: np.ndarray, reshaped: np.ndarray, backbone: Backbone) -> np.ndarray:
    # Mirrors ssfl.models.uses_flat_features: MLP consumes the flat 115-vector, CNN/LSTM the
    # Eq.19 (23,5) reshape. Duplicated (not imported) so ssfl.data never depends on ssfl.models.
    return flat if backbone == Backbone.mlp else reshaped


class TensorFeatureDataset(Dataset):
    """Wraps flat+reshaped feature arrays (and optional labels) as a torch ``Dataset``, picking
    whichever representation the requested backbone consumes."""

    def __init__(
        self,
        features_flat: np.ndarray,
        features_reshaped: np.ndarray,
        backbone: Backbone,
        labels: np.ndarray | None = None,
    ) -> None:
        x = _select_features(features_flat, features_reshaped, backbone)
        self.x = torch.from_numpy(np.ascontiguousarray(x)).float()
        self.y = torch.from_numpy(np.ascontiguousarray(labels)).long() if labels is not None else None

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.x[idx]
        return self.x[idx], self.y[idx]


def load_client_assignments(data_root: Path, scenario: int) -> list[ClientAssignment]:
    payload = json.loads((data_root / "scenarios" / f"{scenario}.json").read_text())
    return [
        ClientAssignment(
            client_id=c["client_id"],
            device_id=c["device_id"],
            scenario=payload["scenario"],
            class_local_indices={int(k): v for k, v in c["class_local_indices"].items()},
        )
        for c in payload["clients"]
    ]


def load_client_private_data(
    data_root: Path, assignment: ClientAssignment, backbone: Backbone
) -> TensorFeatureDataset:
    flat_parts, reshaped_parts, label_parts = [], [], []
    for label, local_indices in sorted(assignment.class_local_indices.items()):
        npz = np.load(data_root / "private" / f"{assignment.device_id}_{label}.npz")
        idx = np.array(local_indices, dtype=np.int64)
        flat_parts.append(npz["features_flat"][idx])
        reshaped_parts.append(npz["features_reshaped"][idx])
        label_parts.append(np.full(len(idx), label, dtype=np.int64))
    flat = np.concatenate(flat_parts, axis=0)
    reshaped = np.concatenate(reshaped_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    return TensorFeatureDataset(flat, reshaped, backbone, labels)


def load_open_data(data_root: Path, backbone: Backbone) -> TensorFeatureDataset:
    flat = np.load(data_root / "open" / "features.npy")
    reshaped = np.load(data_root / "open" / "features_reshaped.npy")
    return TensorFeatureDataset(flat, reshaped, backbone, labels=None)


def load_test_data(data_root: Path, backbone: Backbone) -> TensorFeatureDataset:
    flat = np.load(data_root / "test" / "features.npy")
    reshaped = np.load(data_root / "test" / "features_reshaped.npy")
    labels = np.load(data_root / "test" / "labels.npy")
    return TensorFeatureDataset(flat, reshaped, backbone, labels)
