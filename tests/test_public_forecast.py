"""Training contracts: train-only transforms, authenticated adapters, finite metrics."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mr_nwm.public_forecast import (
    ForecastScaler,
    adapter_state,
    build_forecaster,
    physical_metrics,
    restore_adapters,
    select_indices,
    train_forecaster,
)


def split(level=1.0):
    history = np.full((4, 8, 4), level, dtype=np.float32)
    return SimpleNamespace(
        history=history,
        observed=np.ones_like(history, dtype=bool),
        targets=np.full((4, 5), level, dtype=np.float32),
        sample_ids=[f"sample-{i}" for i in range(4)],
    )


def test_scaler_uses_training_values_and_masks_only():
    train = split()
    train.history[0, 0, 1] = 1e9
    train.observed[0, 0, 1] = False
    scaler = ForecastScaler.fit(train)
    assert scaler.mean[1] == 1.0
    before = scaler.to_dict()
    heldout = scaler.transform(split(100))
    assert scaler.to_dict() == before
    assert heldout["context"][0, 1] > 100
    train_encoded = scaler.transform(train)
    assert train_encoded["context"][0, 1] == 0
    np.testing.assert_allclose(scaler.inverse(heldout["targets"]), 100, rtol=1e-5)


def test_physical_metrics_independent_identity_and_fail_closed():
    truth = np.array([[1.0, 2.0], [3.0, 4.0]])
    pred = truth + 1
    score = physical_metrics(pred, truth)
    assert score["mae"] == 1
    assert score["rmse"] == 1
    assert score["nmse_db"] == pytest.approx(10 * np.log10(4 / 30))
    assert score["horizon_mae"] == [1, 1]
    pred[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        physical_metrics(pred, truth)


def test_evaluation_selection_is_stable_under_source_order_changes():
    names = ["a", "b", "c", "d"]
    first = {names[i] for i in select_indices(names, 2)}
    names.reverse()
    assert first == {names[i] for i in select_indices(names, 2)}


def test_adapters_exclude_backbone_and_reject_incomplete_checkpoint():
    model = torch.nn.Module()
    model.backbone = torch.nn.Module()
    model.backbone.language_model = torch.nn.Linear(2, 2)
    model.head = torch.nn.Linear(2, 1)
    saved = adapter_state(model)
    assert set(saved) == {"head.weight", "head.bias"}
    restore_adapters(model, saved)
    with pytest.raises(ValueError, match="adapter"):
        restore_adapters(model, {"head.weight": saved["head.weight"]})


def test_optimizer_and_sampler_resume_match_uninterrupted_training(tmp_path):
    data = split()
    data.history[..., 0] = np.arange(32).reshape(4, 8) / 10 + 1
    scaler = ForecastScaler.fit(data)
    encoded = scaler.transform(data)
    config = {
        "seed": 11,
        "updates": 4,
        "batch_size": 2,
        "learning_rate": 0.001,
        "validation_every": 1,
    }
    indices = np.arange(4)
    torch.manual_seed(11)
    complete = build_forecaster("numeric")
    train_forecaster(
        complete, encoded, encoded, indices, scaler, config, tmp_path / "complete", "contract"
    )
    torch.manual_seed(11)
    partial = build_forecaster("numeric")
    train_forecaster(
        partial,
        encoded,
        encoded,
        indices,
        scaler,
        config,
        tmp_path / "partial",
        "contract",
        stop_after=2,
    )
    resumed = build_forecaster("numeric")
    train_forecaster(
        resumed, encoded, encoded, indices, scaler, config, tmp_path / "partial", "contract"
    )
    for key, value in adapter_state(complete).items():
        torch.testing.assert_close(value, adapter_state(resumed)[key], rtol=0, atol=0)
