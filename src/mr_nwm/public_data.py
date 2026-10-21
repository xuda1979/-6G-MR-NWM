"""Authenticated measured traces, causal windows and immutable group holdouts.

This observational forecast task has no fabricated radio-control actions. The
loader returns physical units; any fitted normalization belongs to the trainer.
"""

import calendar
import csv
import hashlib
import io
import json
import os
import tempfile
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .public_sources import (
    CICV_FILES,
    CICV_REVISION,
    UCC4G_SHA256,
    UCC4G_URL,
    UCC5G_REVISION,
    UCC5G_SHA256,
    UCC5G_URL,
)

FEATURE_NAMES = ["past_target", "rsrp_dbm", "snr_db", "speed_m_per_s"]


@dataclass
class Trace:
    name: str
    times: np.ndarray
    values: np.ndarray
    source: str
    audit: dict = field(default_factory=dict)


@dataclass
class PublicSplit:
    history: np.ndarray
    observed: np.ndarray
    targets: np.ndarray
    sample_ids: list[str]
    group_ids: list[str]
    periods: np.ndarray


@dataclass
class PublicDataset:
    train: PublicSplit
    validation: PublicSplit
    test: PublicSplit
    manifest: dict
    feature_names: list[str]
    target_unit: str


def fetch_verified(url, path, *, sha256=None, git_sha1=None, size=None):
    """Never use a stale/tampered cached source; atomically cache fresh bytes."""
    if sha256 is None and git_sha1 is None:
        raise ValueError("a pinned source identity is required")
    path = Path(path)
    cached = path.exists()
    if cached:
        content = path.read_bytes()
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "MR-NWM-public-data/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read(16 * 1024 * 1024 + 1)
        if len(content) > 16 * 1024 * 1024:
            raise ValueError("source exceeds declared download size ceiling")
    digest = hashlib.sha256(content).hexdigest()
    blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
    if (
        (sha256 is not None and digest != sha256)
        or (git_sha1 is not None and blob != git_sha1)
        or (size is not None and len(content) != size)
    ):
        raise ValueError(f"source integrity mismatch: {path.name}")
    if not cached:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, pending = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(pending, path)
        finally:
            Path(pending).unlink(missing_ok=True)
    return content


def _number(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return np.nan
    return result if np.isfinite(result) and abs(result) < np.finfo(np.float32).max else np.nan


def _clean_trace(name, times, values, source, raw_rows):
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1, 4)
    finite_time = np.isfinite(times)
    invalid_times = int((~finite_time).sum())
    times, values = times[finite_time], values[finite_time]
    order = np.argsort(times, kind="stable")
    times, values = times[order], values[order]
    # Keep the last logger row at a repeated timestamp; do not average observations.
    keep = np.r_[times[1:] != times[:-1], True] if len(times) else np.array([], dtype=bool)
    duplicate_times = int(len(times) - keep.sum())
    times, values = times[keep], values[keep]
    values[values[:, 0] < 0, 0] = np.nan
    values[values[:, 3] < 0, 3] = np.nan
    return Trace(
        name,
        times,
        values,
        source,
        {
            "raw_rows": raw_rows,
            "accepted_rows": len(times),
            "invalid_timestamps": invalid_times,
            "duplicate_timestamps": duplicate_times,
            "missing_values": int((~np.isfinite(values)).sum()),
        },
    )


def parse_ucc_csv(name, content):
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    times, values = [], []
    for row in rows:
        try:
            timestamp = calendar.timegm(
                datetime.strptime(row["Timestamp"], "%Y.%m.%d_%H.%M.%S").timetuple()
            )
        except (ValueError, KeyError, TypeError):
            timestamp = np.nan
        times.append(timestamp)
        values.append(
            [
                _number(row.get("DL_bitrate")) / 1000.0,
                _number(row.get("RSRP")),
                _number(row.get("SNR")),
                _number(row.get("Speed")) / 3.6,
            ]
        )
    return _clean_trace(name, times, values, "ucc", len(rows))


def parse_cicv_txt(name, content):
    lines = content.decode("utf-8-sig").splitlines()
    nonempty = [line.split() for line in lines if line.strip()]
    if not nonempty:
        raise ValueError(f"empty trace: {name}")
    header = nonempty[0]
    required = ["pub_time(ms)", "delay(ms)", "rsrp(db)", "sinr(db)", "velocity(m/s)"]
    if any(column not in header for column in required):
        raise ValueError(f"unsupported CICV header: {name}")
    indices = [header.index(column) for column in required]
    times, values = [], []
    malformed = 0
    for row in nonempty[1:]:
        if len(row) <= max(indices):
            malformed += 1
            continue
        selected = [_number(row[index]) for index in indices]
        times.append(selected[0] / 1000)
        values.append(selected[1:])
    result = _clean_trace(name, times, values, "cicv5g", len(nonempty) - 1)
    result.audit["malformed_rows"] = malformed
    return result


def read_source_traces(name, cache_dir):
    cache = Path(cache_dir)
    if name in {"ucc4g", "ucc5g"}:
        url, identity = (UCC4G_URL, UCC4G_SHA256) if name == "ucc4g" else (UCC5G_URL, UCC5G_SHA256)
        raw = fetch_verified(url, cache / f"{name}.zip", sha256=identity)
        traces, files = [], []
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = sorted(
                path
                for path in archive.namelist()
                if path.lower().endswith(".csv") and not path.startswith("__MACOSX/")
            )
            for path in names:
                content = archive.read(path)
                item = parse_ucc_csv(path, content)
                traces.append(item)
                files.append(
                    {"path": path, "sha256": hashlib.sha256(content).hexdigest(), **item.audit}
                )
        return traces, {
            "url": url,
            "sha256": identity,
            "bytes": len(raw),
            "revision": "zenodo.1219679-v1" if name == "ucc4g" else UCC5G_REVISION,
            "license": "CC-BY-4.0" if name == "ucc4g" else "GPL-3.0 repository license",
            "license_url": "https://zenodo.org/api/records/1219679"
            if name == "ucc4g"
            else f"https://github.com/uccmisl/5Gdataset/blob/{UCC5G_REVISION}/LICENSE",
            "transforms": ["DL_bitrate: kbps / 1000 -> Mbps", "Speed: km/h / 3.6 -> m/s"],
            "files": files,
        }
    if name != "cicv5g":
        raise ValueError(f"unknown public dataset: {name}")
    base = f"https://raw.githubusercontent.com/zxr805/CICV5G/{CICV_REVISION}/"
    by_blob = {entry[1]: entry for entry in CICV_FILES}

    def download(entry):
        path, identity, size = entry
        url = base + urllib.parse.quote(path)
        return identity, fetch_verified(
            url, cache / "cicv5g" / (identity + ".txt"), git_sha1=identity, size=size
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        contents = dict(pool.map(download, by_blob.values()))
    traces, files = [], []
    for path, identity, size in CICV_FILES:
        content = contents[identity]
        item = parse_cicv_txt(path, content)
        traces.append(item)
        files.append(
            {
                "path": path,
                "git_sha1": identity,
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": size,
                **item.audit,
            }
        )
    return traces, {
        "url": "https://github.com/zxr805/CICV5G",
        "revision": CICV_REVISION,
        "license": "CC-BY-4.0 data; article has separate publication terms",
        "license_url": "https://zenodo.org/api/records/17475688",
        "transforms": [
            "pub_time(ms) / 1000 -> seconds",
            "delay remains ms",
            "velocity remains m/s",
        ],
        "files": files,
    }


def _fingerprint(trace):
    values = np.nan_to_num(trace.values, nan=0.0).astype("<f8")
    payload = (
        trace.times.astype("<f8").tobytes() + values.tobytes() + np.isfinite(trace.values).tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def split_for_group(name):
    """Frozen name-hash rule, independent of values, targets and training seed."""
    bucket = int(hashlib.sha256(("mr-nwm-public-v1:" + name).encode()).hexdigest(), 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 80 else "test"


def prepare_dataset(name, traces, *, history=8, horizon=5, stride=5, source=None):
    if any(not isinstance(value, int) or value <= 0 for value in (history, horizon, stride)):
        raise ValueError("history, horizon and stride must be positive integers")
    if history < 2:
        raise ValueError("history must contain at least two observations to estimate cadence")
    bins = {key: [] for key in ("train", "validation", "test")}
    counts = {
        "input_traces": len(traces),
        "duplicate_traces": 0,
        "gap_boundaries": 0,
        "rejected_missing_targets": 0,
        "rejected_initial_target": 0,
    }
    fingerprints, retained = {}, []
    for trace in sorted(traces, key=lambda item: item.name):
        if trace.values.shape != (len(trace.times), 4) or not np.isfinite(trace.times).all():
            raise ValueError("trace schema mismatch")
        if len(trace.times) > 1 and not (np.diff(trace.times) > 0).all():
            raise ValueError("trace timestamps must be unique and increasing")
        fingerprint = _fingerprint(trace)
        if fingerprint in fingerprints:
            counts["duplicate_traces"] += 1
            continue
        fingerprints[fingerprint] = trace.name
        retained.append(trace)
    # Identical source events appearing in partial/aliased runs must remain in
    # one split even when the complete trace hashes differ.
    parent = {trace.name: trace.name for trace in retained}

    def representative(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    event_owner = {}
    for trace in retained:
        for index, time in enumerate(trace.times):
            row = trace.values[index]
            event = hashlib.sha256(
                np.asarray([time], dtype="<f8").tobytes()
                + np.nan_to_num(row, nan=0.0).astype("<f8").tobytes()
                + np.isfinite(row).tobytes()
            ).digest()
            previous = event_owner.setdefault(event, trace.name)
            a, b = representative(previous), representative(trace.name)
            if a != b:
                parent[max(a, b)] = min(a, b)
    groups = {trace.name: representative(trace.name) for trace in retained}
    for trace in retained:
        differences = np.diff(trace.times)
        if not len(differences):
            continue
        period = float(np.median(differences))
        boundaries = np.flatnonzero(differences > 1.5 * period) + 1
        counts["gap_boundaries"] += len(boundaries)
        split = split_for_group(groups[trace.name])
        for left, right in zip(np.r_[0, boundaries], np.r_[boundaries, len(trace.times)]):
            for start in range(int(left), int(right) - history - horizon + 1, stride):
                context = trace.values[start : start + history]
                target = trace.values[start + history : start + history + horizon, 0]
                if not np.isfinite(target).all() or (target < 0).any():
                    counts["rejected_missing_targets"] += 1
                    continue
                if not np.isfinite(context[-1, 0]) or context[-1, 0] < 0:
                    counts["rejected_initial_target"] += 1
                    continue
                mask = np.isfinite(context)
                mask[:, 0] &= context[:, 0] >= 0
                values = np.where(mask, context, 0).astype(np.float32)
                y = target.astype(np.float32)
                if not np.isfinite(values).all() or not np.isfinite(y).all():
                    raise ValueError("float32 data overflow")
                sample_id = f"{name}:{trace.name}:{start}:{history}:{horizon}"
                # Input cadence must be independent of future target timestamps.
                history_period = float(np.median(np.diff(trace.times[start : start + history])))
                bins[split].append(
                    (values, mask.copy(), y, sample_id, groups[trace.name], history_period)
                )
    splits = {}
    for key, entries in bins.items():
        if not entries:
            raise ValueError(f"empty public-data split: {key}; need more independent traces")
        splits[key] = PublicSplit(
            np.stack([row[0] for row in entries]),
            np.stack([row[1] for row in entries]),
            np.stack([row[2] for row in entries]),
            [row[3] for row in entries],
            [row[4] for row in entries],
            np.asarray([row[5] for row in entries], dtype=np.float32),
        )
    group_sets = [set(value.group_ids) for value in splits.values()]
    disjoint = all(
        not first & second for i, first in enumerate(group_sets) for second in group_sets[i + 1 :]
    )
    if not disjoint:
        raise ValueError("group split leakage")
    manifest = {
        "protocol": "public-measured-traces-v1",
        "dataset": name,
        "history": history,
        "horizon": horizon,
        "stride": stride,
        "input_period": "median timestamp difference in each historical window only",
        "gap_filter_period": "whole-trace median used only for offline segmentation",
        "features": FEATURE_NAMES,
        "target_unit": "ms" if name == "cicv5g" else "Mbps",
        "source": source or {},
        "counts": {
            **counts,
            "unique_traces": len(retained),
            "unique_groups": len(set(groups.values())),
        },
        "split_rule": (
            "sha256(mr-nwm-public-v1:+canonical_group_name) modulo 100: "
            "<70 train, <80 validation, else test; overlap-connected traces share a group"
        ),
        "trace_groups": groups,
        "trace_content_sha256": fingerprints,
        "split_audit": {
            "groups_disjoint": disjoint,
            "exact_event_overlap_grouped": True,
            "normalization": "none; physical units returned",
        },
        "splits": {
            key: {
                "windows": len(value.targets),
                "groups": sorted(set(value.group_ids)),
                "sample_ids_sha256": hashlib.sha256(
                    "\n".join(value.sample_ids).encode()
                ).hexdigest(),
            }
            for key, value in splits.items()
        },
    }
    # Assert JSON safety at the producer boundary, not only when saving a report.
    json.dumps(manifest, allow_nan=False)
    return PublicDataset(
        **splits,
        manifest=manifest,
        feature_names=FEATURE_NAMES.copy(),
        target_unit=manifest["target_unit"],
    )


def load_public_dataset(name, cache_dir, history=8, horizon=5, stride=5):
    traces, source = read_source_traces(name, cache_dir)
    return prepare_dataset(
        name, traces, history=history, horizon=horizon, stride=stride, source=source
    )
