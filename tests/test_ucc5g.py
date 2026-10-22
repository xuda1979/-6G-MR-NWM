import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from mr_nwm.ucc5g import (
    COLUMNS,
    build_windows,
    load_ucc5g_sessions,
    segment_session,
)


def synthetic_session(rows=600, seed=3, start_epoch=1_000_000):
    """A compact UCC 5G stand-in: 1 Hz KPIs, one gap, one GPS glitch."""
    rng = np.random.default_rng(seed)
    table = np.empty((rows, 10), dtype=np.float64)
    for i in range(rows):
        table[i, 0] = float(start_epoch + i)
    table[:, 1] = -8.4 + 0.001 * np.sin(np.arange(rows) / 20.0)  # longitude
    table[:, 2] = 51.8 + 0.001 * np.cos(np.arange(rows) / 25.0)  # latitude
    table[:, 3] = 40.0  # speed km/h
    dl = 2000.0 + 400.0 * np.sin(np.arange(rows) / 8.0) + rng.normal(0, 100, rows)
    table[:, 4] = np.clip(dl, 0, None)  # DL
    table[:, 5] = -90.0 + rng.normal(0, 2, rows)  # RSRP
    table[:, 6] = -12.0 + rng.normal(0, 1, rows)  # RSRQ
    table[:, 7] = 2.0 + rng.normal(0, 1, rows)  # SNR
    table[:, 8] = 10.0  # UL
    table[:, 9] = np.nan  # PINGAVG missing by default
    table[50, 4] = np.nan  # missing downlink
    table[60:70, 6:8] = np.nan  # missing RSRQ/SNR
    table[100, 2] = 29.0  # GPS glitch row
    # three disjoint stretches separated by 120 s gaps
    table[200:, 0] += 120.0
    table[400:, 0] += 120.0
    return table


def datetime_from_epoch(epoch):
    import datetime as datetime_module

    return datetime_module.datetime.fromtimestamp(epoch)


def write_session_csv(path, table):
    """Write one UCC 5G-shaped session CSV with the full 26-column header."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        for row in table:
            timestamp = datetime_str(row[0])
            values = [""] * len(COLUMNS)

            def put(name, value):
                values[COLUMNS.index(name)] = "" if np.isnan(value) else f"{value:.10g}"

            put("Timestamp", 0.0)  # placeholder replaced below
            values[COLUMNS.index("Timestamp")] = timestamp
            put("Longitude", row[1])
            put("Latitude", row[2])
            put("Speed", row[3])
            put("DL_bitrate", row[4])
            put("RSRP", row[5])
            put("RSRQ", row[6])
            put("SNR", row[7])
            put("UL_bitrate", row[8])
            put("PINGAVG", row[9])
            writer.writerow(values)
    return path


def datetime_str(epoch):
    import datetime as datetime_module

    return datetime_module.datetime.fromtimestamp(epoch).strftime("%Y.%m.%d_%H.%M.%S")


def build_sessions_dir(tmp_path, count=3, rows=300):
    """Write ``count`` synthetic session CSVs under a nested directory tree."""
    root = tmp_path / "ucc5g"
    (root / "Netflix" / "Driving").mkdir(parents=True)
    (root / "Download" / "Static").mkdir(parents=True)
    paths = []
    for index in range(count):
        if index % 2 == 0:
            folder = root / "Netflix" / "Driving"
        else:
            folder = root / "Download" / "Static"
        table = synthetic_session(rows=rows, seed=3 + index, start_epoch=1_000_000 + index * 10_000)
        path = folder / f"B_2019.12.{index + 1:02d}_10.00.00.csv"
        write_session_csv(path, table)
        paths.append(path)
    return root, paths


def test_load_rejects_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ucc5g_sessions(tmp_path / "nowhere", expect_files=3, expect_rows=90)


def test_load_rejects_wrong_file_count(tmp_path):
    root, _ = build_sessions_dir(tmp_path, count=2)
    with pytest.raises(ValueError, match="CSV files"):
        load_ucc5g_sessions(root, expect_files=3, expect_rows=900)


def test_load_rejects_wrong_row_count(tmp_path):
    root, _ = build_sessions_dir(tmp_path, count=3)
    with pytest.raises(ValueError, match="rows"):
        load_ucc5g_sessions(root, expect_files=3, expect_rows=100)


def test_load_rejects_bad_header(tmp_path):
    root, paths = build_sessions_dir(tmp_path, count=3)
    paths[0].write_text("Wrong,Header\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        load_ucc5g_sessions(root, expect_files=3, expect_rows=900)


def test_load_rejects_unparseable_timestamp(tmp_path):
    root, paths = build_sessions_dir(tmp_path, count=3)
    lines = paths[0].read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace(lines[1].split(",")[0], "not-a-timestamp", 1)
    paths[0].write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="timestamp"):
        load_ucc5g_sessions(root, expect_files=3, expect_rows=900)


def test_load_parses_synthetic_sessions(tmp_path):
    root, _ = build_sessions_dir(tmp_path, count=3, rows=300)
    sessions = load_ucc5g_sessions(root, expect_files=3, expect_rows=900)
    assert len(sessions) == 3
    assert all(table.shape == (300, 10) for _, table in sessions)
    first = sessions[0][1]
    assert np.isnan(first[50, 4])  # missing downlink preserved as NaN
    assert np.isfinite(first[:, 0]).all()  # timestamps all parse
    assert abs(first[0, 2] - 51.8) < 0.01  # Cork-ish latitude


def test_segmentation_splits_on_gap_and_glitch():
    table = synthetic_session()
    segments = segment_session(table)
    assert 2 <= len(segments) <= 6
    bounds = {s for pair in segments for s in pair}
    assert 0 in bounds and len(table) in bounds
    starts = [s for s, _ in segments]
    assert any(int(table[s, 0]) == int(table[200, 0]) for s in starts if s > 0)
    glitch_segments = [e - s for s, e in segments if s == 100]
    assert glitch_segments == [1]


def test_windows_are_session_isolated_and_chronological(tmp_path):
    root, _ = build_sessions_dir(tmp_path, count=3, rows=300)
    sessions = load_ucc5g_sessions(root, expect_files=3, expect_rows=900)
    windows = build_windows(sessions, history=8, horizon=4)
    assert windows.history.shape[1:] == (8, 4)
    assert windows.future.shape[1:] == (4, 4)
    assert windows.context.shape[1] == 5
    assert set(windows.splits.unique().tolist()) == {0, 1, 2}
    counts = windows.stats["window_counts"]
    assert counts["train"] + counts["val"] + counts["test"] == len(windows.splits)
    starts = {index: entry["start_timestamp"] for index, entry in windows.session_table.items()}
    train_max = max(starts[i] for i, e in windows.session_table.items() if e["split"] == "train")
    val_min = min(starts[i] for i, e in windows.session_table.items() if e["split"] == "val")
    test_min = min(starts[i] for i, e in windows.session_table.items() if e["split"] == "test")
    assert train_max < val_min < test_min


def test_masks_track_missing_observations(tmp_path):
    root, _ = build_sessions_dir(tmp_path, count=3, rows=300)
    sessions = load_ucc5g_sessions(root, expect_files=3, expect_rows=900)
    windows = build_windows(sessions, history=8, horizon=4)
    assert windows.history_mask.dtype == torch.bool
    assert windows.future_mask.dtype == torch.bool
    covering = (windows.history_mask[:, :, 0] == False).any(dim=1)  # noqa: E712
    assert covering.any()
    assert torch.isfinite(windows.history).all()
    assert torch.isfinite(windows.future).all()


def test_normalization_uses_training_rows_only(tmp_path):
    root, _ = build_sessions_dir(tmp_path, count=3, rows=300)
    sessions = load_ucc5g_sessions(root, expect_files=3, expect_rows=900)
    windows = build_windows(sessions, history=8, horizon=4)
    history, future, context = windows.normalized()
    train = windows.splits == 0
    assert (
        torch.allclose(history[train].mean(), torch.zeros(4), atol=1e-4)
        or history[train].mean().abs().max() < 1e-3
    )
    assert history.shape == windows.history.shape
    assert context.shape == windows.context.shape


def test_build_windows_rejects_bad_arguments(tmp_path):
    root, _ = build_sessions_dir(tmp_path, count=3, rows=300)
    sessions = load_ucc5g_sessions(root, expect_files=3, expect_rows=900)
    with pytest.raises(ValueError, match="positive"):
        build_windows(sessions, history=0, horizon=4)
    with pytest.raises(ValueError, match="fractions"):
        build_windows(sessions, history=8, horizon=4, train_fraction=1.5)
    with pytest.raises(ValueError, match="positive"):
        build_windows(sessions, history=8, horizon=4, val_fraction=0.9)


def test_real_dataset_loader_contract():
    """The packaged loader contract: correct on-disk dataset parses end to end.

    Skipped when the 25 MB dataset has not been downloaded, so CI without the
    data still passes.
    """
    root = Path("data/ucc5g/5G-production-dataset")
    if not root.exists():
        pytest.skip("UCC 5G dataset not downloaded")
    sessions = load_ucc5g_sessions(root)
    assert len(sessions) == 83
    total = sum(table.shape[0] for _, table in sessions)
    assert total == 188_711
    windows = build_windows(sessions, history=8, horizon=4)
    assert windows.kpi_dim == 4
    assert windows.context_dim == 5
    assert windows.session_ids.unique().numel() == len(windows.session_table)
    assert set(windows.splits.unique().tolist()) == {0, 1, 2}
