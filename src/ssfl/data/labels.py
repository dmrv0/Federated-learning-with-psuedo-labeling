"""The stable integer label map for N-BaIoT traffic classes.

Label IDs are part of the reproducibility contract — they must never change once artifacts have
been produced under them, so this module is intentionally the single place they are defined.
"""

from __future__ import annotations

LABEL_MAP: dict[str, int] = {
    "benign": 0,
    "gafgyt.combo": 1,
    "gafgyt.junk": 2,
    "gafgyt.scan": 3,
    "gafgyt.tcp": 4,
    "gafgyt.udp": 5,
    "mirai.ack": 6,
    "mirai.scan": 7,
    "mirai.syn": 8,
    "mirai.udp": 9,
    "mirai.udpplain": 10,
}

CLASS_KEY_BY_LABEL: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}

MIRAI_LABELS = frozenset(v for k, v in LABEL_MAP.items() if k.startswith("mirai."))
GAFGYT_LABELS = frozenset(v for k, v in LABEL_MAP.items() if k.startswith("gafgyt."))
BENIGN_LABEL = LABEL_MAP["benign"]

# A device is "6-class" if it has benign + all 5 gafgyt variants but no mirai files at all; every
# other device is expected to be "11-class" (benign + 5 gafgyt + 5 mirai).
SIX_CLASS_KEYS = frozenset({"benign", *(k for k in LABEL_MAP if k.startswith("gafgyt."))})
ELEVEN_CLASS_KEYS = frozenset(LABEL_MAP.keys())

NUM_FEATURES = 115
