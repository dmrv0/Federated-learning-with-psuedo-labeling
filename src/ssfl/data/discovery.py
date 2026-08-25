"""Locate and index N-BaIoT source CSVs on disk.

Handles the flat naming convention actually present in this dataset copy
(``<device_id>.<family>.<attack>.csv``, e.g. ``4.mirai.udp.csv``) and, defensively, the official
UCI nested layout (``<DeviceName>/benign_traffic.csv``, ``<DeviceName>/gafgyt_attacks/combo.csv``,
``<DeviceName>/mirai_attacks/udp.csv``) in case a differently-packaged copy of the archive is
pointed at ``--input`` in the future. This module does no validation of CSV *contents* — see
``ssfl.data.io`` — it only resolves "which file is which (device, class)".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ssfl.data.labels import ELEVEN_CLASS_KEYS, LABEL_MAP, SIX_CLASS_KEYS

_FLAT_RE = re.compile(r"^(?P<device_id>\d+)\.(?P<class_key>.+)\.csv$")

# Official UCI archive layout: "<DeviceName>/benign_traffic.csv",
# "<DeviceName>/gafgyt_attacks/combo.csv", "<DeviceName>/mirai_attacks/udp.csv".
_NESTED_ATTACK_RE = re.compile(r"^(?P<family>gafgyt|mirai)_attacks[/\\](?P<attack>\w+)\.csv$")


class DataDiscoveryError(ValueError):
    """Raised for any structural problem in the discovered file set, with an actionable message."""


@dataclass(frozen=True)
class SourceFile:
    device_id: int
    class_key: str
    label: int
    path: Path


def _parse_flat(path: Path) -> tuple[int, str] | None:
    match = _FLAT_RE.match(path.name)
    if not match or match.group("class_key") not in LABEL_MAP:
        return None
    return int(match.group("device_id")), match.group("class_key")


def _parse_nested(path: Path, device_name_to_id: dict[str, int]) -> tuple[int, str] | None:
    # <DeviceName>/benign_traffic.csv
    if path.name == "benign_traffic.csv" and path.parent.name in device_name_to_id:
        return device_name_to_id[path.parent.name], "benign"
    # <DeviceName>/{gafgyt,mirai}_attacks/<attack>.csv
    rel = f"{path.parent.name}/{path.name}"
    match = _NESTED_ATTACK_RE.match(rel)
    if match and path.parent.parent.name in device_name_to_id:
        class_key = f"{match.group('family')}.{match.group('attack')}"
        if class_key in LABEL_MAP:
            return device_name_to_id[path.parent.parent.name], class_key
    return None


def _read_device_names(input_path: Path) -> dict[str, int]:
    device_info = input_path / "device_info.csv"
    if not device_info.exists():
        return {}
    mapping: dict[str, int] = {}
    with open(device_info, "r") as fh:
        header = fh.readline()
        if "DeviceID" not in header:
            return {}
        for line in fh:
            line = line.strip()
            if not line:
                continue
            device_id_str, device_name = line.split(",", 1)
            mapping[device_name.strip()] = int(device_id_str)
    return mapping


def discover_source_files(input_path: Path) -> list[SourceFile]:
    """Find every ``(device, class)`` CSV under ``input_path``, flat or nested layout.

    Raises ``DataDiscoveryError`` if nothing recognizable is found. Does not validate CSV
    contents or overall device/class completeness — call ``validate_discovery`` for that.
    """
    if not input_path.is_dir():
        raise DataDiscoveryError(f"input path does not exist or is not a directory: {input_path}")

    device_name_to_id = _read_device_names(input_path)
    found: dict[tuple[int, str], Path] = {}
    for path in sorted(input_path.rglob("*.csv")):
        parsed = _parse_flat(path)
        if parsed is None:
            parsed = _parse_nested(path, device_name_to_id)
        if parsed is None:
            continue
        key = parsed
        if key in found:
            raise DataDiscoveryError(
                f"duplicate source file for device={key[0]} class={key[1]!r}: "
                f"{found[key]} and {path}"
            )
        found[key] = path

    if not found:
        raise DataDiscoveryError(
            f"no recognizable N-BaIoT CSVs found under {input_path} "
            "(expected flat '<device>.<family>.<attack>.csv' or nested UCI layout)"
        )

    return [
        SourceFile(device_id=device_id, class_key=class_key, label=LABEL_MAP[class_key], path=path)
        for (device_id, class_key), path in sorted(found.items())
    ]


def validate_discovery(files: list[SourceFile]) -> None:
    """Structural validation only: 9 devices, 7 with all 11 classes, 2 with exactly the 6
    benign+gafgyt classes. Raises ``DataDiscoveryError`` with an actionable message on any
    deviation. Per-file content validation (schema/NaN/row-count) happens in ``ssfl.data.io``.
    """
    by_device: dict[int, set[str]] = {}
    for f in files:
        by_device.setdefault(f.device_id, set()).add(f.class_key)

    if len(by_device) != 9:
        raise DataDiscoveryError(
            f"expected exactly 9 devices, found {len(by_device)}: {sorted(by_device)}"
        )

    eleven_class_devices = []
    six_class_devices = []
    for device_id, class_keys in sorted(by_device.items()):
        if class_keys == ELEVEN_CLASS_KEYS:
            eleven_class_devices.append(device_id)
        elif class_keys == SIX_CLASS_KEYS:
            six_class_devices.append(device_id)
        else:
            missing = ELEVEN_CLASS_KEYS - class_keys
            extra = class_keys - ELEVEN_CLASS_KEYS
            raise DataDiscoveryError(
                f"device {device_id} has an unrecognized class set "
                f"(missing={sorted(missing)}, unexpected={sorted(extra)}); "
                "expected either all 11 classes or exactly the 6 benign+gafgyt classes"
            )

    if len(eleven_class_devices) != 7 or len(six_class_devices) != 2:
        raise DataDiscoveryError(
            f"expected 7 eleven-class devices and 2 six-class devices, got "
            f"{len(eleven_class_devices)} eleven-class {eleven_class_devices} and "
            f"{len(six_class_devices)} six-class {six_class_devices}"
        )
