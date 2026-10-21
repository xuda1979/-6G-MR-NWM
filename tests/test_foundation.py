from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mr_nwm.data import generate_batch
from mr_nwm.foundation import FoundationInputs, LanguageBasedWorldModel
from mr_nwm.modalities import ModalityObservation


class ToyCausalDecoder(nn.Module):
    """Offline API contract fixture; not presented as a pretrained language model."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(64, 24)
        self.projection = nn.Linear(24, 24)
        self.calls = 0

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask, position_ids, **kwargs):
        self.calls += 1
        observed = inputs_embeds * attention_mask[..., None]
        hidden = observed.cumsum(1) / (position_ids + 1)[..., None]
        return SimpleNamespace(last_hidden_state=self.projection(hidden).tanh())


def fixture():
    model = LanguageBasedWorldModel(ToyCausalDecoder(), modality_dimensions={"csi": 4, "graph": 3})
    nn.init.normal_(model.delta.weight, std=0.03)
    batch = generate_batch(2, 10, seed=77)
    observation = ModalityObservation(torch.randn(2, 3, 4), torch.zeros(2, 3), 0.001)
    inputs = FoundationInputs(
        torch.tensor([[1, 3, 2], [1, 4, 0]]),
        torch.tensor([[1, 1, 1], [1, 1, 0]]),
        {"csi": observation},
    )
    return model, batch, inputs


def test_language_backbone_runs_slow_only_and_shares_physics():
    model, batch, inputs = fixture()
    model.eval()
    packet = model.forward_slow_tier(inputs, batch.initial, batch.context)
    assert model.backbone.language_model.calls == 2
    memory, state = None, batch.initial
    for index in range(5):
        state, _, memory = model.forward_fast_tier_step(
            state, batch.actions[:, index], packet, memory
        )
    assert model.backbone.language_model.calls == 2
    assert model.fast_physics_core is model.backbone.physics
    assert torch.isfinite(state).all()


def test_language_preferences_do_not_change_physics_but_evidence_does():
    model, batch, inputs = fixture()
    model.eval()
    changed = FoundationInputs(63 - inputs.intent_ids, inputs.intent_mask, inputs.modalities)
    first = model.rollout(
        inputs, batch.initial, batch.actions, batch.context, macro_period_seconds=0.005
    )
    second = model.rollout(
        changed, batch.initial, batch.actions, batch.context, macro_period_seconds=0.005
    )
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    changed.evidence_ids = torch.ones(2, 2, dtype=torch.long)
    changed.evidence_mask = torch.ones(2, 2, dtype=torch.long)
    third = model.rollout(
        changed, batch.initial, batch.actions, batch.context, macro_period_seconds=0.005
    )
    assert not torch.allclose(first, third)


def test_multimodal_changes_reach_prediction_and_frozen_lm_preserves_input_gradients():
    model, batch, inputs = fixture()
    prediction = model(inputs, batch.initial, batch.actions[:, :5], batch.context)["pred_states"]
    prediction.square().mean().backward()
    assert all(parameter.grad is None for parameter in model.backbone.language_model.parameters())
    adapter = model.backbone.modalities.adapters["csi"][1].weight
    assert adapter.grad is not None and adapter.grad.abs().sum() > 0
    assert model.modulator.weight.grad.abs().sum() > 0
    before = prediction.detach()
    inputs.modalities["csi"].values = inputs.modalities["csi"].values + 0.7
    # The numeric adapter must retain absolute levels instead of normalizing them away.
    after = model(inputs, batch.initial, batch.actions[:, :5], batch.context)["pred_states"]
    assert not torch.allclose(before, after)


def test_missing_modalities_and_stale_observation_age_are_explicit():
    model, batch, inputs = fixture()
    baseline = model.forward_slow_tier(inputs, batch.initial, batch.context)["physical_context"]
    inputs.modalities["csi"].age_seconds += 0.5
    aged = model.forward_slow_tier(inputs, batch.initial, batch.context)["physical_context"]
    assert not torch.allclose(baseline, aged)
    inputs.modalities["csi"].age_seconds.fill_(-0.1)
    with pytest.raises(ValueError, match="future observations"):
        model.forward_slow_tier(inputs, batch.initial, batch.context)
    inputs.modalities = {}
    assert torch.isfinite(
        model.forward_slow_tier(inputs, batch.initial, batch.context)["gamma"]
    ).all()


def test_graph_edges_and_masked_missing_nodes():
    model, batch, inputs = fixture()
    values = torch.randn(2, 3, 3)
    values[:, 2] = float("nan")
    inputs.modalities = {
        "graph": ModalityObservation(
            values,
            torch.zeros(2, 3),
            0.1,
            valid=torch.tensor([[True, True, False], [True, True, False]]),
            adjacency=torch.ones(2, 3, 3),
        )
    }
    assert torch.isfinite(
        model.forward_slow_tier(inputs, batch.initial, batch.context)["gamma"]
    ).all()


def test_input_budget_prevents_unbounded_modal_prefix():
    model, batch, inputs = fixture()
    model.backbone.max_tokens = 4
    with pytest.raises(ValueError, match="max_tokens"):
        model.forward_slow_tier(inputs, batch.initial, batch.context)
    assert model.backbone.language_model.calls == 0


def test_semantic_budget_is_checked_before_either_decoder_pass():
    model, batch, inputs = fixture()
    model.backbone.max_tokens = 8  # physical prefix fits; semantic prefix does not
    with pytest.raises(ValueError, match="max_tokens"):
        model.forward_slow_tier(inputs, batch.initial, batch.context)
    assert model.backbone.language_model.calls == 0


def test_masked_text_ids_are_never_looked_up():
    model, batch, inputs = fixture()
    model.eval()
    baseline = model.forward_slow_tier(inputs, batch.initial, batch.context)
    inputs.intent_ids[~inputs.intent_mask.bool()] = -100
    masked = model.forward_slow_tier(inputs, batch.initial, batch.context)
    torch.testing.assert_close(baseline["objective_weights"], masked["objective_weights"])
    inputs.intent_ids[0, 0] = -100
    with pytest.raises((IndexError, RuntimeError)):
        model.forward_slow_tier(inputs, batch.initial, batch.context)


def test_multi_window_forecast_ages_snapshot_and_preserves_caller_inputs():
    model, batch, inputs = fixture()
    model.eval()
    with pytest.raises(ValueError, match="macro_period_seconds"):
        model.rollout(inputs, batch.initial, batch.actions, batch.context)
    assert model.backbone.language_model.calls == 0
    age = inputs.modalities["csi"].age_seconds.clone()
    first = model(inputs, batch.initial, batch.actions[:, :5], batch.context)
    second = model(
        inputs.advance(0.5),
        first["pred_states"][:, -1],
        batch.actions[:, 5:],
        batch.context,
        first["sync_summary"],
    )
    frozen = model(
        inputs,
        first["pred_states"][:, -1],
        batch.actions[:, 5:],
        batch.context,
        first["sync_summary"],
    )
    actual = model.rollout(
        inputs, batch.initial, batch.actions, batch.context, macro_period_seconds=0.5
    )
    torch.testing.assert_close(actual[:, :5], first["pred_states"])
    torch.testing.assert_close(actual[:, 5:], second["pred_states"])
    assert not torch.allclose(actual[:, 5:], frozen["pred_states"])
    torch.testing.assert_close(inputs.modalities["csi"].age_seconds, age)


@pytest.mark.parametrize("period", [0, -1, float("nan"), float("inf")])
def test_invalid_macro_clock_rejected(period):
    model, batch, inputs = fixture()
    with pytest.raises(ValueError, match="macro_period_seconds"):
        model.rollout(
            inputs, batch.initial, batch.actions, batch.context, macro_period_seconds=period
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_actual_transformer_forward_without_network_download(dtype):
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    config._attn_implementation = "eager"
    decoder = transformers.Qwen2Model(config)
    model = (
        LanguageBasedWorldModel(decoder, modality_dimensions={"csi": 4, "graph": 3})
        .to(dtype=dtype)
        .eval()
    )
    nn.init.normal_(model.delta.weight, std=0.03)
    batch = generate_batch(2, 5, 70)
    inputs = FoundationInputs(
        torch.ones(2, 3, dtype=torch.long),
        torch.ones(2, 3, dtype=torch.long),
        {
            "csi": ModalityObservation(torch.randn(2, 3, 4), torch.full((2, 3), 86400.0), 0.001),
            "graph": ModalityObservation(
                torch.randn(2, 2, 3),
                torch.zeros(2, 2),
                1.0,
                adjacency=torch.ones(2, 2, 2) * 1e-8,
            ),
        },
    )
    output = model(
        inputs, batch.initial.to(dtype), batch.actions.to(dtype), batch.context.to(dtype)
    )
    assert output["pred_states"].shape == (2, 5, 6)
    assert torch.isfinite(output["objective_weights"]).all()
    assert torch.isfinite(output["pred_states"]).all()
    output["pred_states"].float().square().mean().backward()
    gradient = model.backbone.modalities.adapters["csi"][1].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert all(parameter.grad is None for parameter in decoder.parameters())
