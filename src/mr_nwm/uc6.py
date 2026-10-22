"""Loader and forecasting task for the 5G-IANA UC6 real 5G measurement campaign.

The dataset (Zenodo record 10037164, CC BY 4.0, MD5 f8047eb5a2d57a5f789363b0c93757e5)
contains 26,733 one-second KPI records collected between a Nokia edge server and a
vehicle-mounted OBU near Ulm: Timestamp, Latitude, Longitude, Throughput(Mbps),
RTT(msec), RSRQ(db), RSRP(dbm). RSRQ/RSRP are missing in 7,310 rows; throughput in
21 rows; coordinates glitch far from the route. There are no control-action logs,
so this is an observation-forecasting task: past KPI plus mobility context predict
future throughput, RTT and signal strength. Windows are built on the native one-
second physical grid; overlapping windows are never split randomly across train and
evaluation. Normalization statistics are computed only on the training segments.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

COLUMNS = (
    "Timestamp",
    "Latitude",
    "Longitude",
    "Throughput(Mbps)",
    "RTT(msec)",
    "RSRQ(db)",
    "RSRP(dbm)",
)
KPI_NAMES = ("throughput_mbps", "rtt_ms", "rsrq_db", "rsrp_dbm")
EXPECTED_MD5 = "f8047eb5a2d57a5f789363b0c93757e5"
EXPECTED_ROWS = 26_733
_METERS_PER_DEG_LAT = 111_320.0
_METERS_PER_DEG_LON = 111_320.0 * np.cos(np.radians(48.4))


@dataclass(frozen=True)
class UC6Windows:
    """Windowed tensors plus the provenance needed to audit the split."""

    history: torch.Tensor  # float32 [N, history, 4] raw KPI
    future: torch.Tensor  # float32 [N, horizon, 4] raw KPI
    context: torch.Tensor  # float32 [N, 5] mean mobility context of the window
    history_mask: torch.Tensor  # bool [N, history, 4] observed-before-interpolation
    future_mask: torch.Tensor  # bool [N, horizon, 4]
    segment_ids: torch.Tensor  # int64 [N]
    splits: torch.Tensor  # int64 [N] 0=train 1=val 2=test, by whole segment
    stats: dict  # normalization provenance, fit on training windows only
    segment_table: dict  # segment index -> rows/seconds/split provenance

    @property
    def kpi_dim(self):
        return self.history.shape[-1]

    @property
    def context_dim(self):
        return self.context.shape[-1]

    def rows_for_split(self, split):
        return torch.nonzero(self.splits == split).squeeze(1)

    def normalized(self):
        """Return (history, future, context) normalized with training statistics."""
        mean = torch.tensor(self.stats["kpi_mean"])
        std = torch.tensor(self.stats["kpi_std"])
        context_mean = torch.tensor(self.stats["context_mean"])
        context_std = torch.tensor(self.stats["context_std"])
        history = (self.history - mean) / std
        future = (self.future - mean) / std
        context = (self.context - context_mean) / context_std
        return history, future, context


def load_uc6_csv(path):
    """Parse and strictly validate the UC6 CSV; return a float64 table in file order."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"UC6 CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if tuple(header) != COLUMNS:
            raise ValueError(f"unexpected UC6 header: {header}")
        rows = [row for row in reader if row]
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} UC6 rows, found {len(rows)}")
    table = np.empty((len(rows), len(COLUMNS)), dtype=np.float64)
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            table[i, j] = np.nan if cell == "" else float(cell)
    if not np.isfinite(table[:, 0]).all():
        raise ValueError("UC6 timestamps must all be present")
    return table


def segment_uc6(table, *, gap_seconds=60.0, position_jump_meters=250.0):
    """Split rows into trajectories on large time gaps or GPS position glitches.

    Rows sharing one timestamp stay together. A position jump larger than
    ``position_jump_meters`` between consecutive rows starts a new segment so
    GPS glitches cannot weld disjoint route stretches into one trajectory.
    """
    timestamps = table[:, 0]
    latitude, longitude = table[:, 1], table[:, 2]
    start = 0
    segments = []
    for i in range(1, len(table)):
        dt = timestamps[i] - timestamps[i - 1]
        jump = np.hypot(
            (latitude[i] - latitude[i - 1]) * _METERS_PER_DEG_LAT,
            (longitude[i] - longitude[i - 1]) * _METERS_PER_DEG_LON,
        )
        if dt > gap_seconds or jump > position_jump_meters:
            segments.append((start, i))
            start = i
    segments.append((start, len(table)))
    return segments


def _clean_position(table, segments, radius_deg=0.02):
    """Replace within-segment coordinate glitches with the segment's route median.

    Real measurements 2.2 km or more off the segment median latitude are treated
    as receiver glitches; replacing them keeps the mobility context usable
    without discarding the KPI rows they carry.
    """
    cleaned = table.copy()
    latitude, longitude = cleaned[:, 1], cleaned[:, 2]
    for start, end in segments:
        lat_med = float(np.nanmedian(latitude[start:end]))
        far = np.abs(latitude[start:end] - lat_med) > radius_deg
        latitude[start:end][far] = lat_med
        longitude[start:end][far] = float(np.nanmedian(longitude[start:end]))
    return cleaned


def _interpolate_column(values):
    """Linear interpolation over missing values on the one-second grid, in place."""
    good = np.isfinite(values)
    if good.all():
        return values
    if not good.any():
        values[:] = 0.0
        return values
    indices = np.arange(len(values))
    values[~good] = np.interp(indices[~good], indices[good], values[good])
    return values


def _segment_speed_mps(timestamps, latitude, longitude):
    distance = np.hypot(
        np.diff(latitude, prepend=latitude[0]) * _METERS_PER_DEG_LAT,
        np.diff(longitude, prepend=longitude[0]) * _METERS_PER_DEG_LON,
    )
    dt = np.diff(timestamps, prepend=timestamps[0])
    speed = np.where(dt > 0, distance / np.maximum(dt, 1.0), 0.0)
    return np.clip(speed, 0.0, 83.0)  # plausible ceiling: 300 km/h


def build_windows(
    table,
    segments,
    *,
    history=8,
    horizon=4,
    train_fraction=0.7,
    val_fraction=0.15,
):
    """Build history/future windows on the native one-second grid.

    Splitting is by whole trajectory segment, in chronological order: the
    earliest 70% of usable segments train, the next 15% validate, and the
    latest 15% test on genuinely later measurement sessions. Overlapping
    windows therefore never cross the train/evaluation boundary. Segments
    shorter than one full window are skipped. Returns a ``UC6Windows``.
    """
    if history < 1 or horizon < 1:
        raise ValueError("history and horizon must be positive")
    if not 0 < train_fraction < 1 or not 0 <= val_fraction < 1:
        raise ValueError("split fractions must be within (0, 1)")
    if train_fraction + val_fraction >= 1:
        raise ValueError("test fraction must stay positive")
    usable = [(s, e) for s, e in segments if e - s >= history + horizon + 1]
    if len(usable) < 3:
        raise ValueError(f"only {len(usable)} usable segments; need at least 3")
    usable.sort(key=lambda se: table[se[0], 0])  # chronological session order
    # Floor-based counts that still guarantee every split keeps at least one
    # whole segment, so the smallest usable datasets remain auditable.
    n_train = max(1, min(int(train_fraction * len(usable)), len(usable) - 2))
    n_val = max(1, min(int(val_fraction * len(usable)), len(usable) - n_train - 1))
    assignments = ["train"] * n_train + ["val"] * n_val + ["test"] * (len(usable) - n_train - n_val)

    history_blocks, future_blocks, context_blocks = [], [], []
    history_mask_blocks, future_mask_blocks, segment_ids, split_ids = [], [], [], []
    segment_table = {}
    split_code = {"train": 0, "val": 1, "test": 2}
    for segment_index, ((start, end), split) in enumerate(zip(usable, assignments)):
        block = table[start:end]
        timestamps = block[:, 0]
        latitude, longitude = block[:, 1], block[:, 2]
        raw_kpi = np.stack([block[:, 3], block[:, 4], block[:, 5], block[:, 6]], axis=1)
        observed = np.isfinite(raw_kpi)
        kpi = raw_kpi.copy()
        for column in range(kpi.shape[1]):
            _interpolate_column(kpi[:, column])
        speed = _segment_speed_mps(timestamps, latitude, longitude)
        heading = np.arctan2(
            np.diff(longitude, prepend=longitude[0]),
            np.diff(latitude, prepend=latitude[0]),
        )
        context = np.stack([latitude, longitude, np.sin(heading), np.cos(heading), speed], axis=1)
        segment_table[segment_index] = {
            "rows": int(end - start),
            "start_timestamp": int(timestamps[0]),
            "end_timestamp": int(timestamps[-1]),
            "seconds": int(timestamps[-1] - timestamps[0] + 1),
            "split": split,
        }
        for i in range(0, len(block) - history - horizon + 1):
            history_blocks.append(kpi[i : i + history])
            future_blocks.append(kpi[i + history : i + history + horizon])
            context_blocks.append(context[i : i + history].mean(axis=0))
            history_mask_blocks.append(observed[i : i + history])
            future_mask_blocks.append(observed[i + history : i + history + horizon])
            segment_ids.append(segment_index)
            split_ids.append(split_code[split])
    history_t = torch.from_numpy(np.stack(history_blocks)).float()
    future_t = torch.from_numpy(np.stack(future_blocks)).float()
    context_t = torch.from_numpy(np.stack(context_blocks)).float()
    history_mask = torch.from_numpy(np.stack(history_mask_blocks))
    future_mask = torch.from_numpy(np.stack(future_mask_blocks))
    segment_ids_t = torch.tensor(segment_ids, dtype=torch.int64)
    splits_t = torch.tensor(split_ids, dtype=torch.int64)
    stats = _fit_stats(history_t, context_t, splits_t)
    return UC6Windows(
        history=history_t,
        future=future_t,
        context=context_t,
        history_mask=history_mask,
        future_mask=future_mask,
        segment_ids=segment_ids_t,
        splits=splits_t,
        stats=stats,
        segment_table=segment_table,
    )


def _fit_stats(history, context, splits):
    """Normalization statistics fit only on training rows."""
    train = splits == 0
    if train.sum() == 0:
        raise ValueError("no training windows after segment split")
    kpi_mean = history[train].reshape(-1, history.shape[-1]).mean(dim=0)
    kpi_std = history[train].reshape(-1, history.shape[-1]).std(dim=0).clamp_min(1e-6)
    context_mean = context[train].mean(dim=0)
    context_std = context[train].std(dim=0).clamp_min(1e-6)
    return {
        "kpi_mean": kpi_mean.tolist(),
        "kpi_std": kpi_std.tolist(),
        "context_mean": context_mean.tolist(),
        "context_std": context_std.tolist(),
        "window_counts": {
            "train": int((splits == 0).sum()),
            "val": int((splits == 1).sum()),
            "test": int((splits == 2).sum()),
        },
    }
