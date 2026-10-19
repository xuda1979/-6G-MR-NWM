"""Strict comparison of deterministic benchmark evidence, including vector metrics."""

import math
from numbers import Real


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


def validate_result_rows(manifest):
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
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise RuntimeError("Missing, duplicated, or unexpected variant/seed result rows")
