"""Independent acceptance checks for public-trace forecasting and recovery."""

import copy
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mr_nwm import public_forecast as forecast


def script_module(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"independent_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_split():
    rng = np.random.default_rng(918)
    history = rng.uniform(1, 4, (20, 8, 4)).astype(np.float32)
    targets = history[:, -1, :1] + np.arange(1, 6, dtype=np.float32)[None] * 0.1
    return SimpleNamespace(
        history=history,
        observed=np.ones_like(history, dtype=bool),
        targets=targets,
        periods=np.ones(20, dtype=np.float32),
        sample_ids=[f"independent-{i}" for i in range(20)],
    )


def test_targets_cannot_change_forecast_or_ridge_output():
    split = fixture_split()
    scaler = forecast.ForecastScaler.fit(split)
    encoded = scaler.transform(split)
    altered = {**encoded, "targets": encoded["targets"] + 1e6}
    model = forecast.build_forecaster("numeric").eval()
    with torch.no_grad():
        model.delta.weight.normal_()
        a = forecast.predict_tensor(model, encoded, np.arange(5), 1.0)
        b = forecast.predict_tensor(model, altered, np.arange(5), 1.0)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    ridge_a, _ = forecast.ridge_forecast(encoded, encoded)
    ridge_b, _ = forecast.ridge_forecast(encoded, altered)
    np.testing.assert_array_equal(ridge_a, ridge_b)


def test_interruption_resume_matches_uninterrupted_optimizer_and_sampler(tmp_path, monkeypatch):
    split = fixture_split()
    scaler = forecast.ForecastScaler.fit(split)
    encoded = scaler.transform(split)
    config = {
        "seed": 71,
        "learning_rate": 0.001,
        "updates": 4,
        "batch_size": 4,
        "validation_every": 1,
    }
    torch.manual_seed(101)
    complete = forecast.build_forecaster("numeric")
    expected = forecast.train_forecaster(
        complete,
        encoded,
        encoded,
        np.arange(6),
        scaler,
        config,
        tmp_path / "complete",
        "independent-same-contract",
    )
    torch.manual_seed(101)
    partial = forecast.build_forecaster("numeric")
    actual_save = forecast.save_checkpoint

    def interrupted_save(path, value):
        actual_save(path, value)
        if value.get("step") == 2:
            raise InterruptedError("independent simulated process termination")

    monkeypatch.setattr(forecast, "save_checkpoint", interrupted_save)
    with pytest.raises(InterruptedError):
        forecast.train_forecaster(
            partial,
            encoded,
            encoded,
            np.arange(6),
            scaler,
            config,
            tmp_path / "partial",
            "independent-same-contract",
        )
    monkeypatch.setattr(forecast, "save_checkpoint", actual_save)
    torch.manual_seed(999)
    resumed = forecast.build_forecaster("numeric")
    result = forecast.train_forecaster(
        resumed,
        encoded,
        encoded,
        np.arange(6),
        scaler,
        config,
        tmp_path / "partial",
        "independent-same-contract",
    )
    assert result["selected_step"] == expected["selected_step"]
    assert result["validation_mae"] == expected["validation_mae"]
    assert result["trace"] == expected["trace"]
    expected_state = forecast.adapter_state(complete)
    actual_state = forecast.adapter_state(resumed)
    for key in expected_state:
        torch.testing.assert_close(expected_state[key], actual_state[key], rtol=0, atol=0)


def test_resume_rejects_changed_contract(tmp_path):
    split = fixture_split()
    scaler = forecast.ForecastScaler.fit(split)
    encoded = scaler.transform(split)
    config = {
        "seed": 71,
        "learning_rate": 0.001,
        "updates": 1,
        "batch_size": 4,
        "validation_every": 1,
    }
    model = forecast.build_forecaster("numeric")
    forecast.train_forecaster(
        model,
        encoded,
        encoded,
        np.arange(6),
        scaler,
        config,
        tmp_path,
        "original",
    )
    with pytest.raises(ValueError, match="identity"):
        forecast.train_forecaster(
            model,
            encoded,
            encoded,
            np.arange(6),
            scaler,
            config,
            tmp_path,
            "tampered",
        )


@pytest.fixture
def public_evidence(tmp_path, monkeypatch):
    verifier = script_module("verify_public")
    split = fixture_split()
    data = SimpleNamespace(
        train=split,
        validation=split,
        test=split,
        manifest={"fixture": "independent authenticated-evidence fixture"},
    )
    monkeypatch.setattr(verifier, "load_public_dataset", lambda *args, **kwargs: data)
    scaler = forecast.ForecastScaler.fit(split)
    settings = {
        "dataset": "fixture",
        "history": 8,
        "horizon": 5,
        "stride": 5,
        "test_limit": 16,
        "models": ["numeric", "foundation"],
        "seeds": [11, 22, 33],
    }
    indices = forecast.select_indices(split.sample_ids, 16)
    truth = split.targets[indices].astype(np.float64)
    prediction = truth + 0.5
    metrics = {
        "mae": 0.5,
        "rmse": 0.5,
        "nmse_db": float(10 * np.log10(np.sum((prediction - truth) ** 2) / np.sum(truth**2))),
        "horizon_mae": [0.5] * 5,
    }
    rows = []
    combinations = [("persistence", None), ("ridge", None)] + [
        (name, seed) for name in settings["models"] for seed in settings["seeds"]
    ]
    identity = forecast.contract_hash(
        {
            "settings": settings,
            "data": data.manifest,
            "sources": {},
            "environment": {},
        }
    )
    for model, seed in combinations:
        file = tmp_path / f"{model}_{seed}_predictions.npz"
        np.savez_compressed(
            file,
            prediction=prediction,
            target=truth,
            sample_ids=np.asarray([split.sample_ids[i] for i in indices]),
        )
        row = {
            "model": model,
            "seed": seed,
            "test": copy.deepcopy(metrics),
            "predictions": {
                "path": file.name,
                "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            },
        }
        if seed is not None:
            checkpoint = tmp_path / f"{model}_{seed}.pt"
            torch.save(
                {
                    "identity_sha256": identity,
                    "model": model,
                    "seed": seed,
                    "scaler": scaler.to_dict(),
                    "selected_step": 1,
                    "nominal_period_seconds": 1.0,
                },
                checkpoint,
            )
            row["training"] = {"selected_step": 1, "nominal_period_seconds": 1.0}
            row["checkpoint"] = {
                "path": checkpoint.name,
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
        rows.append(row)
    manifest = {
        "complete": True,
        "settings": settings,
        "data": data.manifest,
        "source_sha256": {},
        "environment": {},
        "identity_sha256": identity,
        "scaler": scaler.to_dict(),
        "evaluation": {
            "test_sample_ids": [split.sample_ids[i] for i in indices],
        },
        "runs": rows,
    }
    (tmp_path / "metrics.json").write_text(json.dumps(manifest))
    return verifier, tmp_path, manifest


def test_evidence_verifier_accepts_complete_fixture(public_evidence):
    verifier, directory, _ = public_evidence
    assert verifier.verify(directory, directory, replay=False)["result"] == "PASS"


@pytest.mark.parametrize(
    "mutation", ["duplicate", "nan", "tamper", "incomplete", "missing_checkpoint"]
)
def test_evidence_verifier_rejects_invalid_records(public_evidence, mutation):
    verifier, directory, manifest = public_evidence
    if mutation == "duplicate":
        manifest["runs"][-1] = copy.deepcopy(manifest["runs"][-2])
    elif mutation == "nan":
        manifest["runs"][0]["test"]["mae"] = float("nan")
    elif mutation == "tamper":
        file = directory / manifest["runs"][0]["predictions"]["path"]
        file.write_bytes(file.read_bytes() + b"tampered")
    elif mutation == "missing_checkpoint":
        del manifest["runs"][-1]["checkpoint"]
    else:
        manifest["complete"] = False
    (directory / "metrics.json").write_text(json.dumps(manifest))
    with pytest.raises((ValueError, AssertionError)):
        verifier.verify(directory, directory, replay=False)


def test_independent_scorer_rejects_nonfinite_truth():
    verifier = script_module("verify_public")
    truth = np.array([[1.0, float("nan")]])
    with pytest.raises(ValueError):
        verifier.independently_score(np.ones_like(truth), truth)


def test_report_rejects_three_rows_of_the_same_initialization(public_evidence, monkeypatch):
    verifier, directory, manifest = public_evidence
    reporter = script_module("report_public")
    monkeypatch.setattr(reporter, "verify", verifier.verify)
    root = directory / "reports"
    root.mkdir()
    artifacts = list(directory.glob("*.npz")) + list(directory.glob("*.pt"))
    manifest["runs"][-1] = copy.deepcopy(manifest["runs"][-2])
    for name in ["ucc4g", "ucc5g", "cicv5g"]:
        target = root / name
        target.mkdir()
        for artifact in artifacts:
            shutil.copy2(artifact, target / artifact.name)
        (target / "metrics.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicated"):
        reporter.summarize(root)
