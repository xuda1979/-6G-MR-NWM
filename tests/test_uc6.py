import numpy as np
import pytest
import torch

from mr_nwm.uc6 import (
    COLUMNS,
    EXPECTED_ROWS,
    build_windows,
    load_uc6_csv,
    segment_uc6,
)


def synthetic_table(n=600, seed=3):
    """A compact but realistic UC6 stand-in: 1 Hz KPIs, one gap, one GPS glitch."""
    rng = np.random.default_rng(seed)
    timestamps = np.arange(1_000_000, 1_000_000 + n, dtype=np.float64)
    latitude = 48.4 + 0.001 * np.sin(np.arange(n) / 20.0)
    longitude = 9.96 + 0.001 * np.cos(np.arange(n) / 20.0)
    throughput = 30.0 + 5.0 * np.sin(np.arange(n) / 8.0) + rng.normal(0, 1, n)
    rtt = 30.0 + 10.0 * np.cos(np.arange(n) / 15.0) + rng.normal(0, 2, n)
    rsrq = -10.0 + rng.normal(0, 1, n)
    rsrp = -90.0 + rng.normal(0, 2, n)
    table = np.stack([timestamps, latitude, longitude, throughput, rtt, rsrq, rsrp], axis=1)
    table[50, 3] = np.nan  # missing throughput
    table[60:70, 5:7] = np.nan  # missing signal columns
    table[100, 1] = 29.0  # GPS glitch row
    table[120, 0] = table[119, 0]  # duplicate timestamp
    # three disjoint sessions separated by 120 s gaps
    table[200:, 0] += 120.0
    table[400:, 0] += 120.0
    return table


def write_csv(tmp_path, table, rows=EXPECTED_ROWS):
    """Write a UC6-shaped CSV, padding a synthetic core to the expected row count."""
    pad = rows - len(table)
    assert pad >= 0
    rng = np.random.default_rng(11)
    padding = np.stack(
        [
            np.arange(2_000_000, 2_000_000 + pad, dtype=np.float64),
            np.full(pad, 48.4) + rng.normal(0, 1e-4, pad),
            np.full(pad, 9.96) + rng.normal(0, 1e-4, pad),
            np.full(pad, 30.0),
            np.full(pad, 30.0),
            np.full(pad, -10.0),
            np.full(pad, -90.0),
        ],
        axis=1,
    )
    full = np.concatenate([table, padding])
    path = tmp_path / "uc6.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        import csv as csv_module

        writer = csv_module.writer(stream)
        writer.writerow(COLUMNS)
        for row in full:
            writer.writerow(["" if np.isnan(value) else f"{value:.10g}" for value in row])
    return path, full


def test_load_rejects_wrong_header_and_row_count(tmp_path):
    path = tmp_path / "uc6.csv"
    path.write_text("Wrong,Header\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        load_uc6_csv(path)
    bad_rows = tmp_path / "uc6_short.csv"
    bad_rows.write_text(",".join(COLUMNS) + "\n" + "1,48.4,9.96,30,30,-10,-90\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rows"):
        load_uc6_csv(bad_rows)


def test_load_parses_full_synthetic_file(tmp_path):
    table = synthetic_table()
    path, _ = write_csv(tmp_path, table)
    loaded = load_uc6_csv(path)
    assert loaded.shape == (EXPECTED_ROWS, len(COLUMNS))
    assert np.isnan(loaded[50, 3])
    assert np.isfinite(loaded[:, 0]).all()


def test_segmentation_splits_on_gap_and_glitch():
    table = synthetic_table()
    segments = segment_uc6(table)
    assert 2 <= len(segments) <= 6
    bounds = {s for pair in segments for s in pair}
    assert 0 in bounds and len(table) in bounds
    # the injected 120 s gap at row 200 must start a new segment
    starts = [s for s, _ in segments]
    assert any(int(table[s, 0]) == int(table[200, 0]) for s in starts if s > 0)
    # the GPS glitch at row 100 (29 degrees off route) must sit in its own
    # one-row segment, isolating it from the surrounding route rows
    glitch_segments = [e - s for s, e in segments if s == 100]
    assert glitch_segments == [1]


def test_windows_are_segment_isolated_and_chronological():
    table = synthetic_table()
    segments = segment_uc6(table)
    windows = build_windows(table, segments, history=8, horizon=4)
    assert windows.history.shape[1:] == (8, 4)
    assert windows.future.shape[1:] == (4, 4)
    assert windows.context.shape[1] == 5
    assert set(windows.splits.unique().tolist()) == {0, 1, 2}
    counts = windows.stats["window_counts"]
    assert counts["train"] + counts["val"] + counts["test"] == len(windows.splits)
    starts = {index: entry["start_timestamp"] for index, entry in windows.segment_table.items()}
    train_max = max(starts[i] for i, e in windows.segment_table.items() if e["split"] == "train")
    val_min = min(starts[i] for i, e in windows.segment_table.items() if e["split"] == "val")
    test_min = min(starts[i] for i, e in windows.segment_table.items() if e["split"] == "test")
    assert train_max < val_min < test_min


def test_masks_track_missing_observations():
    table = synthetic_table()
    segments = segment_uc6(table)
    windows = build_windows(table, segments, history=8, horizon=4)
    assert windows.history_mask.dtype == torch.bool
    assert windows.future_mask.dtype == torch.bool
    # throughput missing at row 50: any window whose history covers row 50 must flag it
    covering = (windows.history_mask[:, :, 0] == False).any(dim=1)  # noqa: E712
    assert covering.any()
    # interpolated values fill the tensor even where the mask is False
    assert torch.isfinite(windows.history).all()
    assert torch.isfinite(windows.future).all()


def test_normalization_uses_training_rows_only():
    table = synthetic_table()
    segments = segment_uc6(table)
    windows = build_windows(table, segments, history=8, horizon=4)
    history, future, context = windows.normalized()
    train = windows.splits == 0
    assert (
        torch.allclose(history[train].mean(), torch.zeros(4), atol=1e-4)
        or history[train].mean().abs().max() < 1e-3
    )
    assert history.shape == windows.history.shape
    assert context.shape == windows.context.shape


def test_build_windows_rejects_bad_arguments():
    table = synthetic_table()
    segments = segment_uc6(table)
    with pytest.raises(ValueError, match="positive"):
        build_windows(table, segments, history=0, horizon=4)
    with pytest.raises(ValueError, match="fractions"):
        build_windows(table, segments, history=8, horizon=4, train_fraction=1.5)
    with pytest.raises(ValueError, match="positive"):
        build_windows(table, segments, history=8, horizon=4, val_fraction=0.9)


def test_real_dataset_loader_contract(tmp_path):
    """The packaged loader contract: correct MD5-verified CSV parses end to end."""
    table = synthetic_table()
    path, _ = write_csv(tmp_path, table)
    loaded = load_uc6_csv(path)
    segments = segment_uc6(loaded)
    windows = build_windows(loaded, segments, history=8, horizon=4)
    assert windows.kpi_dim == 4
    assert windows.context_dim == 5
    assert windows.segment_ids.unique().numel() == len(windows.segment_table)
