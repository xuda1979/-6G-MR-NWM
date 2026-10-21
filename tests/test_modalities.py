import math
from dataclasses import replace

import pytest
import torch

from mr_nwm.foundation import FoundationInputs
from mr_nwm.modalities import ModalityObservation, TypedModalityEncoder


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("age", [86400.0, 1e100])
def test_time_features_are_compressed_before_model_precision_cast(dtype, age):
    encoder = TypedModalityEncoder({"csi": 2}, 8).to(dtype=dtype)
    observation = ModalityObservation(
        torch.ones(1, 1, 2), torch.tensor([[age]], dtype=torch.float64), 0.1
    )
    features = []
    encoder.time_embedding.register_forward_pre_hook(lambda _, args: features.append(args[0]))
    tokens, mask = encoder({"csi": observation}, 1, torch.zeros(1, 1, 8, dtype=dtype))
    assert mask.all() and torch.isfinite(tokens).all()
    expected = torch.tensor([[[math.log1p(age), math.log1p(0.1)]]], dtype=dtype)
    torch.testing.assert_close(features[0], expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("scale", [1e-100, 1e100])
def test_graph_normalization_preserves_edges_before_model_precision_cast(dtype, scale):
    encoder = TypedModalityEncoder({"graph": 2}, 8).to(dtype=dtype)
    values = torch.randn(1, 3, 2, requires_grad=True)
    edges = torch.tensor([[[0, 1, 0], [2, 0, 1], [0, 0, 0]]], dtype=torch.float64)
    observation = ModalityObservation(values, torch.zeros(1, 3), 0.1, adjacency=edges)
    reference = torch.zeros(1, 1, 8, dtype=dtype)
    expected, _ = encoder({"graph": observation}, 1, reference)
    actual, _ = encoder({"graph": replace(observation, adjacency=edges * scale)}, 1, reference)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().square().mean().backward()
    assert torch.isfinite(values.grad).all() and values.grad.abs().sum() > 0


def test_unrepresentable_valid_measurements_fail_before_adapter_execution():
    encoder = TypedModalityEncoder({"csi": 2}, 8).half()
    observation = ModalityObservation(torch.full((1, 1, 2), 1e10), torch.zeros(1, 1), 0.1)
    calls = []
    encoder.adapters["csi"].register_forward_pre_hook(lambda *_: calls.append(True))
    with pytest.raises(ValueError, match="representable"):
        encoder({"csi": observation}, 1, torch.zeros(1, 1, 8, dtype=torch.float16))
    assert not calls


@pytest.mark.parametrize("field", ["age_seconds", "adjacency"])
def test_complex_metadata_is_rejected_without_discarding_imaginary_part(field):
    encoder = TypedModalityEncoder({"graph": 2}, 8)
    observation = ModalityObservation(
        torch.ones(1, 1, 2), torch.zeros(1, 1), 0.1, adjacency=torch.ones(1, 1, 1)
    )
    setattr(observation, field, getattr(observation, field).to(torch.complex64) + 1j)
    with pytest.raises(ValueError, match="real"):
        encoder({"graph": observation}, 1, torch.zeros(1, 1, 8))


def test_normalized_graph_messages_are_invariant_to_fractional_edge_scale():
    encoder = TypedModalityEncoder({"graph": 2}, 8)
    values = torch.randn(2, 3, 2, requires_grad=True)
    ages = torch.zeros(2, 3)
    edges = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]]).repeat(2, 1, 1)
    reference = torch.zeros(2, 1, 8)
    observation = ModalityObservation(values, ages, 0.1, adjacency=edges)
    first, _ = encoder({"graph": observation}, 2, reference)
    second, _ = encoder({"graph": replace(observation, adjacency=edges * 0.01)}, 2, reference)
    torch.testing.assert_close(first, second)
    assert torch.isfinite(first).all()  # the isolated third node remains defined
    first.square().mean().backward()
    assert values.grad is not None and values.grad.abs().sum() > 0


def test_expired_nodes_are_masked_and_cannot_send_graph_messages():
    encoder = TypedModalityEncoder({"graph": 2}, 8)
    values = torch.randn(1, 3, 2, requires_grad=True)
    observation = ModalityObservation(
        values,
        torch.tensor([[0.1, 0.2, 0.3]]),
        0.01,
        adjacency=torch.ones(1, 3, 3),
        max_age_seconds=0.2,
    )
    reference = torch.zeros(1, 1, 8)
    baseline, mask = encoder({"graph": observation}, 1, reference)
    assert mask.tolist() == [[True, True, False]]  # exact TTL boundary is still valid
    changed = values.detach().clone()
    changed[:, 2] = float("nan")
    actual, _ = encoder({"graph": replace(observation, values=changed)}, 1, reference)
    torch.testing.assert_close(baseline, actual)
    assert torch.count_nonzero(actual[:, 2]) == 0
    baseline.sum().backward()
    assert torch.count_nonzero(values.grad[:, 2]) == 0
    assert values.grad[:, :2].abs().sum() > 0


def test_snapshot_advance_preserves_missingness_and_expiry_policy():
    observation = ModalityObservation(
        torch.ones(1, 1, 2), torch.zeros(1, 1), 0.01, max_age_seconds=0.1
    )
    inputs = FoundationInputs(
        torch.ones(1, 1, dtype=torch.long), torch.ones(1, 1), {"csi": observation}
    )
    aged = inputs.advance(0.2)
    encoder = TypedModalityEncoder({"csi": 2}, 4)
    tokens, mask = encoder(aged.modalities, 1, torch.zeros(1, 1, 4))
    assert not mask.any() and torch.count_nonzero(tokens) == 0
    assert observation.age_seconds.item() == 0
    assert aged.modalities["csi"].values is observation.values


@pytest.mark.parametrize("max_age", [-1, float("inf"), float("nan")])
def test_invalid_expiry_rejected(max_age):
    encoder = TypedModalityEncoder({"csi": 2}, 4)
    observation = ModalityObservation(
        torch.ones(1, 1, 2), torch.zeros(1, 1), 0.1, max_age_seconds=max_age
    )
    with pytest.raises(ValueError, match="max_age_seconds"):
        encoder({"csi": observation}, 1, torch.zeros(1, 1, 4))
