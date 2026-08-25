# Data Card

## Source

**N-BaIoT** — "Detection of IoT Botnet Attacks", UCI Machine Learning Repository.
http://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT

Creators: Yair Meidan, Michael Bohadana, Yael Mathov, Yisroel Mirsky, Dominik
Breitenbacher, Asaf Shabtai, Yuval Elovici. Donated 2018-03-19.

Real network traffic captured from 9 commercial IoT devices infected by two
IoT botnets (BASHLITE/gafgyt, Mirai). 115 statistical traffic features per
record (`data/features.csv`), computed over multiple decay-window time
scales. Devices 3 and 7 were never infected by Mirai, so they contribute only
benign + 5 gafgyt classes (6 total); the other 7 devices contribute all 11
classes (benign + 5 gafgyt + 5 mirai). On-disk layout used by this project:
89 flat CSVs named `<device_id>.<family>.<attack>.csv`, e.g. `1.benign.csv`,
`4.mirai.udp.csv` (see `data/README.md`, `data/device_info.csv`).

## License

UCI ML Repository datasets are provided for research/educational use; no
separate license file ships with N-BaIoT. This repository's `.gitignore`
excludes `data/` — the raw dataset is never committed, only downloaded
locally per user by whoever reproduces this build.

## Label map

`src/ssfl/data/labels.py` is the single source of truth:

| label | class |
|---:|---|
| 0 | benign |
| 1 | gafgyt.combo |
| 2 | gafgyt.junk |
| 3 | gafgyt.scan |
| 4 | gafgyt.tcp |
| 5 | gafgyt.udp |
| 6 | mirai.ack |
| 7 | mirai.scan |
| 8 | mirai.syn |
| 9 | mirai.udp |
| 10 | mirai.udpplain |

## Transformations (`src/ssfl/data/prepare_data.py`)

1. **Discovery + validation** — 89 source files, 115 named numeric columns
   matching `data/features.csv` exactly, no NaN/Inf, minimum row count per
   file. See `src/ssfl/data/discovery.py`, `src/ssfl/data/io.py`.
2. **Deterministic mini-sampling** — 1,000 rows drawn without replacement per
   (device, class) subset, via `numpy.random.SeedSequence([seed, device_id,
   label])` (entropy-mixed, not summed, so e.g. `(device=2,label=3)` and
   `(device=3,label=2)` never collide). See `src/ssfl/data/sampling.py`.
3. **700 / 100 / 200 split** — one seeded draw per subset sliced directly into
   private (700, index 0:700) / open (100, 700:800) / test (200, 800:1000).
   Disjoint by construction; audited per-row in `audit/source_rows.parquet`.
4. **Min-max scaling** — fit over the full mini-dataset (`all_mini`, the
   canonical scope used everywhere; `private_only` is a secondary fit also
   stored for reference). Constant features are mapped to 0 instead of
   dividing by a zero range. See `src/ssfl/data/scaling.py`.
5. **Equation 19 reshape** — each 115-length feature vector reshaped to a
   `(23, 5)` matrix, `M[row, col] = v[row + 23*col]`, alongside the flat
   115-vector (`features_reshaped` next to `features_flat`/`features.npy`).
6. **Non-IID scenario partitioning** — three client/device layouts (McMahan
   shard partition for scenarios 1/2, Dirichlet(α=0.1) for scenario 3);
   documented as assumption #13 in `REPRODUCIBILITY.md` since the paper names
   the scenarios without specifying the allocation mechanics.

## Splits and record counts (real dataset, seed 2023)

From `artifacts/data/dataset_manifest.json` (`manifest_hash`
`54023e3de197d9dc16f50a89425d5c44bb594486210aacf16ecaa8cccc86ba65`):

| | value |
|---|---:|
| source files | 89 |
| devices | 9 |
| total records | 89,000 |
| private records | 62,300 |
| open records | 8,900 |
| test records | 17,800 |
| scenario 1 clients | 27 |
| scenario 2 clients | 89 |
| scenario 3 clients | 89 |
| normalization mode | all_mini |

## Leakage risks and mitigations

- **Open-set labels.** `open/features.npy` carries no label field and is
  stored in one globally-ordered array (not per-device/per-class files, which
  would leak the true class via filename/path). The only place origin +
  true label are recorded is `audit/source_rows.parquet`, which training and
  federated protocol code must never import — it exists solely for offline
  pseudo-label-accuracy auditing.
- **Private data isolation.** Private splits are stored one file per
  `(device_id, label)` (`private/<device_id>_<label>.npz`); the SSFL protocol
  (M4) must never transmit these arrays, gradients derived directly from
  them, or per-client optimizer state — only labels/metrics cross the wire
  for the open set (see `REPRODUCIBILITY.md` privacy-boundary notes).
- **Test set** is the only split shipped with labels directly
  (`test/labels.npy`), since server-side evaluation legitimately needs them
  and test rows are never used for training or pseudo-labeling.
- **Reproducibility integrity.** `dataset_manifest.json`'s `manifest_hash`
  covers a checksum of every artifact file (via a deterministic, timestamp-free
  `.npz` writer); training code should refuse to run against a data directory
  whose checksums don't match its recorded manifest.

## Known limitations

- Mini-sampling (1,000/subset) is a deliberate reproduction-scale subset of
  the full ~7M-row N-BaIoT dataset, matching the paper's stated experimental
  setup, not the entire corpus.
- Devices 3 and 7 have no mirai classes; scenario partitioning and federated
  aggregation code must tolerate 6-class and 11-class clients side by side.
- `open`/`test` normalization uses a scaler fit over the full mini-dataset
  (`all_mini`); this is not the same distribution the paper's autoencoder
  baseline was fit on, since this project targets multi-class SSFL, not
  the original per-device anomaly-detection setup.
