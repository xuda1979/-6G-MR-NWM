from dataclasses import replace

import pytest
import torch

from mr_nwm.data import generate_batch
from mr_nwm.experts import WirelessSSMExpert
from mr_nwm.model import ModelConfig, MultiRateNestedWorldModel


def initialized(config=ModelConfig()):
    model = MultiRateNestedWorldModel(config)
    # An identity-residual initialization has zero upstream gradients on step zero.
    torch.nn.init.normal_(model.delta.weight, std=0.03)
    return model


def test_sequence_and_step_recurrence_agree_and_memory_matters():
    expert = WirelessSSMExpert(12, 7)
    sequence = torch.randn(2, 4, 12)
    memory, outputs = None, []
    for token in sequence.unbind(1):
        output, memory = expert.step(token, memory)
        outputs.append(output)
    torch.testing.assert_close(expert(sequence), torch.stack(outputs, 1))
    reset, _ = expert.step(sequence[:, -1])
    assert not torch.allclose(reset, outputs[-1])


def test_sharing_is_identity_without_duplicate_checkpoint_keys():
    model = initialized()
    assert model.fast_physics_core is model.backbone.physics
    assert not any("fast_physics_core" in key for key in model.state_dict())
    separate = initialized(replace(ModelConfig(), share_weights=False))
    assert separate.fast_physics_core is not separate.backbone.physics


@pytest.mark.parametrize("modulation", [True, False])
def test_intent_cannot_change_physical_rollout(modulation):
    model = initialized(replace(ModelConfig(), modulation=modulation)).eval()
    batch = generate_batch(3, 10, 17)
    changed = batch.intents.roll(1, 0)
    first = model.rollout(batch.intents, batch.initial, batch.actions, batch.context)
    second = model.rollout(changed, batch.initial, batch.actions, batch.context)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_rollout_advances_state_and_summary_across_macros():
    model = initialized().eval()
    batch = generate_batch(2, 10)
    first = model(batch.intents, batch.initial, batch.actions[:, :5], batch.context)
    second = model(
        batch.intents,
        first["pred_states"][:, -1],
        batch.actions[:, 5:],
        batch.context,
        first["sync_summary"],
    )
    expected = torch.cat([first["pred_states"], second["pred_states"]], 1)
    torch.testing.assert_close(
        model.rollout(batch.intents, batch.initial, batch.actions, batch.context), expected
    )
    reset = model(batch.intents, batch.initial, batch.actions[:, 5:], batch.context)
    assert not torch.allclose(second["pred_states"], reset["pred_states"])


def test_future_actions_cannot_change_earlier_predictions():
    model = initialized().eval()
    batch = generate_batch(2, 10)
    changed = batch.actions.clone()
    changed[:, 3:] = 1 - changed[:, 3:]
    first = model.rollout(batch.intents, batch.initial, batch.actions, batch.context)
    second = model.rollout(batch.intents, batch.initial, changed, batch.context)
    torch.testing.assert_close(first[:, :3], second[:, :3], rtol=0, atol=0)
    assert not torch.allclose(first[:, 3:], second[:, 3:])


def test_cross_scale_gradient_survives_truncation():
    model = initialized(replace(ModelConfig(), truncation=1))
    batch = generate_batch(2, 5)
    output = model(batch.intents, batch.initial, batch.actions, batch.context)
    output["pred_states"][:, -1].square().mean().backward()
    for parameter in (
        model.modulator.weight,
        model.backbone.physics.write.weight,
        model.context_encoder.weight,
    ):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


def test_detached_bridge_is_a_controlled_gradient_cut():
    joint = initialized(replace(ModelConfig(), share_weights=False))
    detached = initialized(replace(joint.config, detach_bridge=True))
    detached.load_state_dict(joint.state_dict())
    batch = generate_batch(2, 5)
    for model in (joint, detached):
        output = model(batch.intents, batch.initial, batch.actions, batch.context)
        output["pred_states"].square().mean().backward()
        assert model.modulator.weight.grad.abs().sum() > 0
        assert model.fast_physics_core.write.weight.grad.abs().sum() > 0
    assert joint.context_encoder.weight.grad.abs().sum() > 0
    assert detached.context_encoder.weight.grad is None


def test_checkpoint_round_trip(tmp_path):
    model = initialized().double().eval()
    batch = generate_batch(2, 10)
    inputs = (batch.intents, batch.initial.double(), batch.actions.double(), batch.context.double())
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = MultiRateNestedWorldModel(model.config).double().eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(model.rollout(*inputs), restored.rollout(*inputs))
    assert restored.fast_physics_core is restored.backbone.physics


@pytest.mark.parametrize("change", [{"horizon": 0}, {"rank": 0}, {"width": 31}, {"truncation": -1}])
def test_invalid_config_rejected(change):
    with pytest.raises(ValueError):
        replace(ModelConfig(), **change)


def test_bad_inputs_rejected():
    model, batch = initialized(), generate_batch(2, 5)
    with pytest.raises(ValueError, match="padding"):
        model(torch.zeros_like(batch.intents), batch.initial, batch.actions, batch.context)
    with pytest.raises(ValueError, match="actions"):
        model(batch.intents, batch.initial, batch.actions[:, :3], batch.context)
    with pytest.raises(ValueError, match="multiple"):
        model.rollout(batch.intents, batch.initial, batch.actions[:, :3], batch.context)
