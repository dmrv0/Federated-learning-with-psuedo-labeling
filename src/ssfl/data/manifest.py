"""Checksums, dataset manifest, allocation statistics, and Figure-2-style allocation plots."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from ssfl.data.partition import ClientAssignment


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str))


def deterministic_savez(path: Path, **arrays: np.ndarray) -> None:
    """``np.savez`` writes a zip archive with per-entry timestamps set to the current time, which
    makes the output file bytes (and therefore its checksum) differ across runs even when the
    array content is identical. This writes the same ``.npz``-compatible zip (readable by
    ``np.load``) with a fixed entry timestamp, so re-running preparation with the same inputs and
    seed produces byte-identical files and a stable ``manifest_hash``."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.save(buf, arrays[name])
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            zf.writestr(info, buf.getvalue())
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def checksum_tree(root: Path) -> dict[str, str]:
    """SHA-256 of every regular file under ``root``, keyed by POSIX path relative to ``root``."""
    checksums: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            checksums[path.relative_to(root).as_posix()] = sha256_file(path)
    return checksums


# ---------------------------------------------------------------------------
# Allocation statistics (Figure 2-style)
# ---------------------------------------------------------------------------


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits, base-2 log, safe against zero entries."""
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def effective_num_classes(counts: np.ndarray) -> float:
    """Hill number of order 1 (exp of Shannon entropy) over a client's class counts."""
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(np.exp(-np.sum(p * np.log(p))))


def gini(counts: np.ndarray) -> float:
    x = np.sort(counts.astype(float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def compute_allocation_stats(assignments: list[ClientAssignment], num_labels: int) -> dict[str, Any]:
    global_counts = np.zeros(num_labels)
    per_client: list[tuple[str, np.ndarray]] = []
    for a in assignments:
        counts = np.zeros(num_labels)
        for label, idxs in a.class_local_indices.items():
            counts[label] = len(idxs)
        global_counts += counts
        per_client.append((a.client_id, counts))

    global_total = global_counts.sum()
    global_p = global_counts / global_total if global_total > 0 else global_counts

    client_stats = []
    for client_id, counts in per_client:
        total = counts.sum()
        p = counts / total if total > 0 else counts
        client_stats.append(
            {
                "client_id": client_id,
                "num_examples": int(total),
                "class_counts": counts.astype(int).tolist(),
                "effective_num_classes": effective_num_classes(counts),
                "js_divergence_from_global": js_divergence(p, global_p) if total > 0 else None,
            }
        )

    sample_counts = np.array([c["num_examples"] for c in client_stats])
    return {
        "num_clients": len(assignments),
        "num_labels": num_labels,
        "global_class_distribution": global_p.tolist(),
        "clients": client_stats,
        "sample_count_gini": gini(sample_counts),
        "sample_count_min": int(sample_counts.min()) if len(sample_counts) else 0,
        "sample_count_max": int(sample_counts.max()) if len(sample_counts) else 0,
        "sample_count_mean": float(sample_counts.mean()) if len(sample_counts) else 0.0,
    }


def plot_allocation(stats: dict[str, Any], scenario: int, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clients = stats["clients"]
    num_labels = stats["num_labels"]
    if not clients:
        return

    counts_matrix = np.array([c["class_counts"] for c in clients], dtype=float)
    totals = counts_matrix.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    proportions = counts_matrix / totals

    fig_height = max(4.0, 0.12 * len(clients))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    y = np.arange(len(clients))
    left = np.zeros(len(clients))
    cmap = plt.get_cmap("tab20", num_labels)
    for label in range(num_labels):
        ax.barh(y, proportions[:, label], left=left, color=cmap(label), label=f"class {label}")
        left += proportions[:, label]

    ax.set_yticks(y)
    ax.set_yticklabels([c["client_id"] for c in clients], fontsize=max(2, min(6, 300 // len(clients))))
    ax.set_xlabel("class proportion")
    ax.set_title(f"Scenario {scenario} client/class allocation")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=6, ncol=1)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
