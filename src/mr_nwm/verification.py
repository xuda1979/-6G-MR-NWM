"""Strict comparison of deterministic benchmark evidence, including vector metrics."""

import math
from dataclasses import asdict
from numbers import Real
from pathlib import Path

import torch

from .provenance import sha256_file


def compare_metrics(actual, expected, tolerance, path="metrics"):
    """Require the complete metric schema and finite values; return scalar count."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be nonnegative and finite")
    if isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise RuntimeError(f"Metric keys differ: {path}")
        return sum(
            compare_metrics(actual[key], expected[key], tolerance, f"{path}.{key}")
            for key in actual
        )
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise RuntimeError(f"Metric lengths differ: {path}")
        return sum(
            compare_metrics(left, right, tolerance, f"{path}[{index}]")
            for index, (left, right) in enumerate(zip(actual, expected))
        )
    if (
        isinstance(actual, bool)
        or isinstance(expected, bool)
        or not isinstance(actual, Real)
        or not isinstance(expected, Real)
    ):
        raise RuntimeError(f"Metric must be numeric: {path}")
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise RuntimeError(f"Non-finite metric: {path}")
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance):
        raise RuntimeError(f"Metric mismatch: {path}: actual={actual}, expected={expected}")
    return 1


def validate_result_rows(manifest, *, complete=True):
    """Missing or duplicated variant/seed rows cannot produce a successful audit."""
    arguments = manifest["arguments"]
    variants, seeds = arguments["variants"], arguments["seeds"]
    if (
        not variants
        or not seeds
        or len(set(variants)) != len(variants)
        or len(set(seeds)) != len(seeds)
    ):
        raise RuntimeError("Manifest requires nonempty, unique variants and seeds")
    expected = {(variant, seed) for variant in variants for seed in seeds}
    actual = [(row["variant"], row["seed"]) for row in manifest["results"]]
    coverage = set(actual) == expected if complete else set(actual) <= expected
    if len(actual) != len(set(actual)) or not coverage:
        raise RuntimeError("Missing, duplicated, or unexpected variant/seed result rows")


def load_verified_checkpoint(model, row, directory, *, require_digest=False):
    """Verify bytes and metadata before loading a run's checkpoint onto the model."""
    if sum(parameter.numel() for parameter in model.parameters()) != row["parameters"]:
        raise RuntimeError(f"Parameter count mismatch: {row['variant']}")
    if not row["parameters"]:
        return
    path = Path(directory) / "checkpoints" / f"{row['variant']}_{row['seed']}.pt"
    expected = row.get("checkpoint_sha256")
    if require_digest and expected is None:
        raise RuntimeError(f"Missing checkpoint digest: {path}; start with --checkpoints")
    if expected is not None and sha256_file(path) != expected:
        raise RuntimeError(f"Checkpoint digest mismatch: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    config = asdict(model.config) if hasattr(model, "config") else None
    if (saved["variant"], saved["seed"], saved["config"]) != (row["variant"], row["seed"], config):
        raise RuntimeError(f"Checkpoint metadata mismatch: {path}")
    model.load_state_dict(saved["state_dict"])
