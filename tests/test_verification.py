from copy import deepcopy

import pytest

from mr_nwm.provenance import source_hashes
from mr_nwm.verification import compare_metrics, validate_result_rows


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_metrics_fail_on_either_side(value):
    for actual, expected in ((value, 1.0), (1.0, value)):
        with pytest.raises(RuntimeError, match="Non-finite"):
            compare_metrics({"nmse_db": actual}, {"nmse_db": expected}, 1e-5)


@pytest.mark.parametrize(
    "actual",
    [{"curve": [1.0, 2.1]}, {"curve": [1.0]}, {"other": [1.0, 2.0]}, {"curve": [True, 2.0]}],
)
def test_vector_corruption_or_missing_metrics_fail(actual):
    with pytest.raises(RuntimeError, match="Metric"):
        compare_metrics(actual, {"curve": [1.0, 2.0]}, 1e-5)


def test_metric_comparison_counts_all_scalar_and_vector_values():
    assert (
        compare_metrics({"mse": 0.1, "curve": [1.0, 2.0]}, {"mse": 0.1, "curve": [1.0, 2.0]}, 0)
        == 3
    )


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
def test_invalid_tolerance_rejected(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        compare_metrics(1.0, 1.0, tolerance)


def test_manifest_cannot_succeed_with_missing_or_duplicated_runs():
    manifest = {
        "arguments": {"variants": ["full", "persistence"], "seeds": [11, 22]},
        "results": [
            {"variant": variant, "seed": seed}
            for variant in ("full", "persistence")
            for seed in (11, 22)
        ],
    }
    validate_result_rows(manifest)
    for results in ([], manifest["results"][:-1], manifest["results"] + [manifest["results"][0]]):
        invalid = deepcopy(manifest)
        invalid["results"] = results
        with pytest.raises(RuntimeError, match="result rows"):
            validate_result_rows(invalid)


def test_source_manifest_excludes_virtualenvs_and_run_outputs(tmp_path):
    for relative in (
        "train_and_eval.py",
        "src/mr_nwm/model.py",
        "tests/test_model.py",
        "scripts/audit.py",
        ".venv/lib/python3.12/site-packages/vendor.py",
        "runs/copied_source.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n")
    sources = source_hashes(tmp_path)
    assert set(sources) == {
        "train_and_eval.py",
        "src/mr_nwm/model.py",
        "tests/test_model.py",
        "scripts/audit.py",
    }
