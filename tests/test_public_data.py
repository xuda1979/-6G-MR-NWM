"""Offline acceptance contracts for measured public-data forecasting inputs."""

import hashlib

import numpy as np
import pytest

from mr_nwm import public_data


def trace(name, *, offset=0.0, count=25):
    times = np.arange(count, dtype=np.float64)
    values = np.column_stack([offset + times, -90 + times / 10, times / 20, np.ones(count)])
    return public_data.Trace(name, times, values, "fixture")


def test_complete_groups_disjoint_and_raw_physical_values_unchanged():
    traces = [trace(f"run-{i:03}", offset=1000 * i) for i in range(45)]
    data = public_data.prepare_dataset("fixture", traces, history=4, horizon=2, stride=2)
    groups = []
    for name in ("train", "validation", "test"):
        split = getattr(data, name)
        assert split.history.shape[1:] == (4, 4)
        assert split.targets.shape[1:] == (2,)
        assert split.observed.dtype == np.bool_
        assert np.isfinite(split.history).all()
        assert np.isfinite(split.targets).all()
        assert len(set(split.sample_ids)) == len(split.sample_ids)
        # A loader must not fit a transform across train and holdout records.
        for group, history in zip(split.group_ids, split.history):
            index = int(group.split("-")[-1])
            assert history[0, 0] >= 1000 * index
        groups.append(set(split.group_ids))
    assert all(groups)
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert data.manifest["split_audit"]["groups_disjoint"]


def test_duplicates_gaps_and_missing_values_cannot_leak():
    traces = [trace(f"run-{i:03}", offset=1000 * i) for i in range(45)]
    source = traces[0]
    source.values[3, 2] = np.nan
    source.values[10, 0] = np.nan
    source.times[15:] += 50
    # Duplicate a source under another name: hash-based content deduplication
    # must happen before names assign train/test membership.
    traces.append(
        public_data.Trace("zz-alias", source.times.copy(), source.values.copy(), "fixture")
    )
    data = public_data.prepare_dataset("fixture", traces, history=4, horizon=2, stride=1)
    manifest = data.manifest
    assert manifest["counts"]["duplicate_traces"] == 1
    assert manifest["counts"]["gap_boundaries"] == 1
    assert manifest["counts"]["rejected_missing_targets"] > 0
    split = next(
        split for split in (data.train, data.validation, data.test) if "run-000" in split.group_ids
    )
    masks = split.observed[np.array(split.group_ids) == "run-000"]
    values = split.history[np.array(split.group_ids) == "run-000"]
    assert (~masks).any()
    assert (values[~masks] == 0).all()
    assert all(s.observed[:, -1, 0].all() for s in (data.train, data.validation, data.test))
    assert not any(
        "zz-alias" == g for s in (data.train, data.validation, data.test) for g in s.group_ids
    )


def test_csv_parse_sorts_deduplicates_and_does_not_read_future_columns():
    raw = (
        "Timestamp,DL_bitrate,RSRP,SNR,Speed,PINGMAX\n"
        "2020.01.01_00.00.02,3000,-93,3,4,9999\n"
        "2020.01.01_00.00.00,1000,-91,1,2,9999\n"
        "2020.01.01_00.00.01,2000,-92,-,3,9999\n"
        "2020.01.01_00.00.01,2200,-92,2,3,9999\n"
    ).encode()
    result = public_data.parse_ucc_csv("run.csv", raw)
    np.testing.assert_array_equal(result.times - result.times[0], [0, 1, 2])
    np.testing.assert_allclose(result.values[:, 0], [1, 2.2, 3])
    assert result.values.shape == (3, 4)
    assert result.audit["duplicate_timestamps"] == 1


def test_cicv_header_schema_and_future_receipt_fields_excluded():
    header = "pub_time(ms) sub_time(ms) delay(ms) velocity(m/s) rsrp(db) sinr(db) currentX(m)\n"
    raw = (header + "1000 1020 20 3 -90 4 999999\n1050 1071 21 4 -91 5 999999\n").encode()
    result = public_data.parse_cicv_txt("run.txt", raw)
    np.testing.assert_allclose(result.times, [1, 1.05])
    np.testing.assert_allclose(result.values, [[20, -90, 4, 3], [21, -91, 5, 4]])


def test_cached_source_is_authenticated_before_use(tmp_path):
    path = tmp_path / "source.zip"
    good = b"trusted source"
    path.write_bytes(b"tampered source")
    expected = hashlib.sha256(good).hexdigest()
    with pytest.raises(ValueError, match="integrity"):
        public_data.fetch_verified("https://example.invalid/source", path, sha256=expected)


def test_partial_duplicate_runs_are_grouped_before_split():
    traces = [trace(f"run-{i:03}", offset=1000 * i) for i in range(45)]
    original = traces[0]
    traces.append(
        public_data.Trace("zz-partial-alias", original.times[:20], original.values[:20], "fixture")
    )
    data = public_data.prepare_dataset("fixture", traces, history=4, horizon=2)
    assert data.manifest["trace_groups"]["zz-partial-alias"] == "run-000"
    locations = {
        key for key in ("train", "validation", "test") if "run-000" in getattr(data, key).group_ids
    }
    assert len(locations) == 1


def test_missing_final_history_target_and_negative_future_are_rejected():
    traces = [trace(f"run-{i:03}", offset=1000 * i) for i in range(45)]
    traces[0].values[3, 0] = np.nan
    traces[0].values[8, 0] = -1
    data = public_data.prepare_dataset("fixture", traces, history=4, horizon=2, stride=1)
    assert data.manifest["counts"]["rejected_initial_target"] > 0
    assert data.manifest["counts"]["rejected_missing_targets"] > 0
    for split in (data.train, data.validation, data.test):
        assert split.observed[:, -1, 0].all()
        assert (split.targets >= 0).all()


def test_future_timestamps_do_not_change_past_inputs_or_period():
    first = [trace(f"run-{i:03}", offset=1000 * i) for i in range(45)]
    second = [trace(f"run-{i:03}", offset=1000 * i) for i in range(45)]
    # Change only timestamps after the first sample's historical window.
    second[0].times[4:] = 3 + 1.1 * np.arange(1, len(second[0].times) - 3)
    datasets = [
        public_data.prepare_dataset("fixture", traces, history=4, horizon=2, stride=1)
        for traces in (first, second)
    ]
    sample = "fixture:run-000:0:4:2"
    views = []
    for data in datasets:
        for key in ("train", "validation", "test"):
            split = getattr(data, key)
            if sample in split.sample_ids:
                index = split.sample_ids.index(sample)
                views.append((split.history[index], split.observed[index], split.periods[index]))
    assert len(views) == 2
    for first_value, second_value in zip(*views):
        np.testing.assert_array_equal(first_value, second_value)
    assert views[0][2] == 1.0


def test_invalid_configuration_and_short_traces_fail_closed():
    with pytest.raises(ValueError):
        public_data.prepare_dataset("fixture", [trace("a")], history=0)
    with pytest.raises(ValueError, match="split"):
        public_data.prepare_dataset("fixture", [trace("a", count=2)], history=8)


def test_loader_uses_offline_fixture_without_network(monkeypatch, tmp_path):
    monkeypatch.setattr(
        public_data,
        "read_source_traces",
        lambda name, cache: (
            [trace(f"run-{i:03}", offset=1000 * i) for i in range(45)],
            {"license": "fixture"},
        ),
    )
    data = public_data.load_public_dataset("ucc4g", tmp_path, history=4, horizon=2)
    assert data.target_unit == "Mbps"
    assert len(data.feature_names) == 4
    assert data.manifest["source"]["license"] == "fixture"
    assert all(len(getattr(data, name).targets) for name in ("train", "validation", "test"))
