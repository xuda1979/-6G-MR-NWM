"""Loader and forecasting task for the UCC 5G production measurement campaign.

The dataset (github.com/uccmisl/5Gdataset, GPL-3.0, ZIP MD5
6a79c2e2a5e343c356bb4243b6355f8a) contains 83 one-second CSV drive-test/session
logs (188,711 rows) recorded in Cork, Ireland on a commercial 5G/LTE network,
one operator ("B"): Timestamp, Longitude, Latitude, Speed, Operatorname, CellID,
NetworkMode, RSRP, RSRQ, SNR, CQI, RSSI, DL_bitrate, UL_bitrate, State, PING*,
CELLHEX, NODEHEX, LACHEX, RAWCELLID, NRxRSRP, NRxRSRQ.

Stage-A task mirrors the UC6 pipeline: past KPI (downlink bitrate, RSRP, RSRQ,
SNR) plus mobility context (position, heading, speed) predict the next
``horizon`` seconds of KPIs. Each CSV file is one measurement session; sessions
are split chronologically into train/val/test so evaluation windows come from
physically later sessions. Missing KPI values are tracked with observation
masks and linearly interpolated on the one-second grid. No control actions are
logged, so this is observation forecasting only.
"""

import csv
import datetime
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

COLUMNS = (
    "Timestamp",
    "Longitude",
    "Latitude",
    "Speed",
    "Operatorname",
    "CellID",
    "NetworkMode",
    "RSRP",
    "RSRQ",
    "SNR",
    "CQI",
    "RSSI",
    "DL_bitrate",
    "UL_bitrate",
    "State",
    "PINGAVG",
    "PINGMIN",
    "PINGMAX",
    "PINGSTDEV",
    "PINGLOSS",
    "CELLHEX",
    "NODEHEX",
    "LACHEX",
    "RAWCELLID",
    "NRxRSRP",
    "NRxRSRQ",
)
KPI_NAMES = ("dl_bitrate_kbps", "rsrp_dbm", "rsrq_db", "snr_db")
EXPECTED_ZIP_MD5 = "6a79c2e2a5e343c356bb4243b6355f8a"
EXPECTED_CSV_FILES = 83
EXPECTED_ROWS = 188_711
_MISSING = ("", "-", "NaN", "nan", "None")

_METERS_PER_DEG_LAT = 111_320.0
_METERS_PER_DEG_LON = 111_320.0 * np.cos(np.radians(51.9))  # Cork latitude


@dataclass(frozen=True)
class UCC5GWindows:
    """Windowed tensors plus the provenance needed to audit the split."""

    history: torch.Tensor  # float32 [N, history, 4] raw KPI
    future: torch.Tensor  # float32 [N, horizon, 4] raw KPI
    context: torch.Tensor  # float32 [N, 5] mean mobility context of the window
    history_mask: torch.Tensor  # bool [N, history, 4] observed-before-interpolation
    future_mask: torch.Tensor  # bool [N, horizon, 4]
    session_ids: torch.Tensor  # int64 [N]
    splits: torch.Tensor  # int64 [N] 0=train 1=val 2=test, by whole session
    stats: dict  # normalization provenance, fit on training windows only
    session_table: dict  # session index -> rows/seconds/split provenance

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


def _parse_timestamp(text):
    """Parse '2019.12.01_12.11.21' into epoch seconds; return NaN on failure."""
    try:
        moment = datetime.datetime.strptime(text, "%Y.%m.%d_%H.%M.%S")
    except (ValueError, TypeError):
        return np.nan
    return moment.timestamp()


def _float_or_nan(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def load_ucc5g_sessions(root, expect_files=EXPECTED_CSV_FILES, expect_rows=EXPECTED_ROWS):
    """Parse every UCC 5G session CSV under ``root``.

    Returns a list of ``(session_name, table)`` where ``table`` is a float64
    array ``[rows, 10]`` in file order with columns:
    timestamp_epoch_s, longitude, latitude, speed_kmh, dl_bitrate_kbps,
    rsrp_dbm, rsrq_db, snr_db, ul_bitrate_kbps, ping_avg_ms.
    Missing/invalid cells become NaN. Sessions whose timestamp column is not
    fully parseable are rejected.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"UCC 5G dataset directory not found: {root}")
    files = sorted(root.rglob("*.csv"))
    if len(files) != expect_files:
        raise ValueError(
            f"expected {expect_files} UCC 5G CSV files under {root}, found {len(files)}"
        )
    sessions = []
    total_rows = 0
    for path in files:
        relative = path.relative_to(root)
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.reader(stream)
            header = next(reader, None)
            if header is None or tuple(header) != COLUMNS:
                raise ValueError(f"unexpected header in {relative}")
            rows = []
            for row in reader:
                if not row:
                    continue
                rows.append(row)
        if not rows:
            raise ValueError(f"empty session file: {relative}")
        table = np.empty((len(rows), 10), dtype=np.float64)
        for i, row in enumerate(rows):
            cells = dict(zip(COLUMNS, row))
            timestamp = _parse_timestamp(cells["Timestamp"])
            if np.isnan(timestamp):
                raise ValueError(f"unparseable timestamp in {relative} row {i}")
            table[i] = (
                timestamp,
                _float_or_nan(cells["Longitude"]),
                _float_or_nan(cells["Latitude"]),
                _float_or_nan(cells["Speed"]),
                _float_or_nan(cells["DL_bitrate"]),
                _float_or_nan(cells["RSRP"]),
                _float_or_nan(cells["RSRQ"]),
                _float_or_nan(cells["SNR"]),
                _float_or_nan(cells["UL_bitrate"]),
                _float_or_nan(cells["PINGAVG"]),
            )
        total_rows += len(rows)
        sessions.append((str(relative), table))
    if total_rows != expect_rows:
        raise ValueError(f"expected {expect_rows} total UCC 5G rows, found {total_rows}")
    return sessions


def segment_session(table, *, gap_seconds=60.0, position_jump_meters=250.0):
    """Split one session's rows into trajectories on time gaps or GPS glitches.

    Rows sharing one timestamp stay together. A position jump larger than
    ``position_jump_meters`` between consecutive rows starts a new segment so
    GPS glitches cannot weld disjoint route stretches into one trajectory.
    """
    timestamps = table[:, 0]
    latitude, longitude = table[:, 2], table[:, 1]
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
    elapsed = np.diff(timestamps, prepend=timestamps[0])
    speed = distance / np.clip(elapsed, 1.0, None)
    return np.nan_to_num(speed, nan=0.0)


def _clean_position(table, segments, radius_deg=0.02):
    """Replace within-segment coordinate glitches with the segment's route median.

    Real measurements 2.2 km or more off the segment median latitude are
    treated as receiver glitches; replacing them keeps the mobility context
    usable without discarding the KPI rows they carry.
    """
    cleaned = table.copy()
    latitude, longitude = cleaned[:, 2], cleaned[:, 1]
    for start, end in segments:
        if not np.isfinite(latitude[start:end]).any():
            continue
        lat_med = float(np.nanmedian(latitude[start:end]))
        far = np.abs(latitude[start:end] - lat_med) > radius_deg
        latitude[start:end][far] = lat_med
        longitude[start:end][far] = float(np.nanmedian(longitude[start:end]))
    return cleaned


def build_windows(
    sessions,
    *,
    history=8,
    horizon=4,
    train_fraction=0.7,
    val_fraction=0.15,
):
    """Build history/future windows on the native one-second grid.

    ``sessions`` is the list produced by :func:`load_ucc5g_sessions`. Each CSV
    file is one measurement session; each session is segmented into
    trajectories, and trajectory segments shorter than one full window are
    skipped. Splitting is by whole session file, in chronological order: the
    earliest 70% of usable sessions train, the next 15% validate, and the
    latest 15% test on genuinely later measurement sessions. Overlapping
    windows therefore never cross the train/evaluation boundary. Returns a
    ``UCC5GWindows``.
    """
    if history < 1 or horizon < 1:
        raise ValueError("history and horizon must be positive")
    if not 0 < train_fraction < 1 or not 0 <= val_fraction < 1:
        raise ValueError("split fractions must be within (0, 1)")
    if train_fraction + val_fraction >= 1:
        raise ValueError("test fraction must stay positive")

    prepared = []
    for name, raw in sessions:
        segments = segment_session(raw)
        cleaned = _clean_position(raw, segments)
        prepared.append((name, cleaned, segments))

    usable = [
        (index, seg)
        for index, (_, _, segments) in enumerate(prepared)
        for seg in segments
        if seg[1] - seg[0] >= history + horizon + 1
    ]
    if len(usable) < 3:
        raise ValueError(f"only {len(usable)} usable segments; need at least 3")
    usable.sort(key=lambda pair: (prepared[pair[0]][1][pair[1][0], 0], pair[0]))
    session_order = sorted(
        {index for index, _ in usable},
        key=lambda index: float(np.nanmin(prepared[index][1][:, 0])),
    )
    n_sessions = len(session_order)
    n_train_sessions = max(1, min(int(train_fraction * n_sessions), n_sessions - 2))
    n_val_sessions = max(1, min(int(val_fraction * n_sessions), n_sessions - n_train_sessions - 1))
    session_split = {}
    for position, index in enumerate(session_order):
        if position < n_train_sessions:
            session_split[index] = "train"
        elif position < n_train_sessions + n_val_sessions:
            session_split[index] = "val"
        else:
            session_split[index] = "test"

    history_blocks, future_blocks, context_blocks = [], [], []
    history_mask_blocks, future_mask_blocks, session_ids, split_ids = [], [], [], []
    session_table = {}
    split_code = {"train": 0, "val": 1, "test": 2}
    session_counter = 0
    for session_index, (name, table, segments) in enumerate(prepared):
        if session_index not in session_split:
            continue
        split = session_split[session_index]
        for seg_start, seg_end in segments:
            if seg_end - seg_start < history + horizon + 1:
                continue
            block = table[seg_start:seg_end]
            timestamps = block[:, 0]
            latitude, longitude = block[:, 2], block[:, 1]
            raw_kpi = np.stack([block[:, 4], block[:, 5], block[:, 6], block[:, 7]], axis=1)
            observed = np.isfinite(raw_kpi)
            kpi = raw_kpi.copy()
            for column in range(kpi.shape[1]):
                _interpolate_column(kpi[:, column])
            speed = _segment_speed_mps(timestamps, latitude, longitude)
            heading = np.arctan2(
                np.diff(longitude, prepend=longitude[0]),
                np.diff(latitude, prepend=latitude[0]),
            )
            context = np.stack(
                [latitude, longitude, np.sin(heading), np.cos(heading), speed], axis=1
            )
            session_table[session_counter] = {
                "session": name,
                "rows": int(seg_end - seg_start),
                "start_timestamp": float(timestamps[0]),
                "end_timestamp": float(timestamps[-1]),
                "seconds": float(timestamps[-1] - timestamps[0] + 1),
                "split": split,
            }
            for i in range(0, len(block) - history - horizon + 1):
                history_blocks.append(kpi[i : i + history])
                future_blocks.append(kpi[i + history : i + history + horizon])
                context_blocks.append(context[i : i + history].mean(axis=0))
                history_mask_blocks.append(observed[i : i + history])
                future_mask_blocks.append(observed[i + history : i + history + horizon])
                session_ids.append(session_counter)
                split_ids.append(split_code[split])
            session_counter += 1

    history_t = torch.from_numpy(np.stack(history_blocks)).float()
    future_t = torch.from_numpy(np.stack(future_blocks)).float()
    context_t = torch.from_numpy(np.stack(context_blocks)).float()
    history_mask = torch.from_numpy(np.stack(history_mask_blocks))
    future_mask = torch.from_numpy(np.stack(future_mask_blocks))
    session_ids_t = torch.tensor(session_ids, dtype=torch.int64)
    splits_t = torch.tensor(split_ids, dtype=torch.int64)
    stats = _fit_stats(history_t, context_t, splits_t)
    return UCC5GWindows(
        history=history_t,
        future=future_t,
        context=context_t,
        history_mask=history_mask,
        future_mask=future_mask,
        session_ids=session_ids_t,
        splits=splits_t,
        stats=stats,
        session_table=session_table,
    )


def _fit_stats(history, context, splits):
    """Fit normalization statistics on training windows only."""
    train = splits == 0
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
