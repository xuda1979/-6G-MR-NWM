import pytest
import torch

from mr_nwm.baselines import Persistence
from mr_nwm.data import generate_batch, transition
from mr_nwm.experiment import counterfactual_audit, nested_loss
from mr_nwm.flow import ConditionalFlowMatchingHead
from mr_nwm.intent import IntentTokenizer, intent_batch
from mr_nwm.metrics import intent_metrics, latency, nmse_db
from mr_nwm.model import MultiRateNestedWorldModel
from mr_nwm.planning import planning_audit


def test_nmse_uses_signal_energy_not_variance():
    target = torch.ones(2, 3)
    assert nmse_db(torch.zeros_like(target), target) == pytest.approx(0)
    assert nmse_db(0.5 * target, target) == pytest.approx(-6.020599913)
    with pytest.raises(ValueError, match="zero-energy"):
        nmse_db(target, torch.zeros_like(target))


def test_intent_metrics_have_interpretable_endpoints():
    weights = torch.tensor([[1.0, 0, 0]])
    assert intent_metrics(weights, weights)["intent_tv"] == 0
    opposite = torch.tensor([[0.0, 1, 0]])
    assert intent_metrics(opposite, weights) == {"intent_tv": 1, "intent_cosine": 0}


def test_ground_truth_is_action_sensitive_and_not_intent_sensitive():
    batch = generate_batch(4, 10)
    low = transition(batch.initial, torch.zeros(4, 2), batch.context)
    high = transition(batch.initial, torch.ones(4, 2), batch.context)
    assert torch.all(high[:, 2:4] <= low[:, 2:4])
    assert torch.all(high[:, 4:] > low[:, 4:])
    again = generate_batch(4, 10)
    torch.testing.assert_close(batch.targets, again.targets, rtol=0, atol=0)


def test_split_and_ood_protocol():
    iid, ood = generate_batch(8, seed=88), generate_batch(8, seed=88, scenario="ood")
    assert ood.context[:, 2].min() > iid.context[:, 2].max()
    assert ood.context[:, 3].min() > iid.context[:, 3].max()
    assert not torch.equal(iid.targets, generate_batch(8, seed=89).targets)
    tokens, weights = intent_batch(8, torch.Generator().manual_seed(2), "composition")
    assert (tokens != 0).any()
    torch.testing.assert_close(weights, torch.full((8, 3), 1 / 3))
    tokenizer = IntentTokenizer()
    train_text = {" ".join(tokenizer.words(text)) for text in ["prioritize throughput"]}
    assert "prioritize throughput" in train_text
    assert tokenizer.encode(["unseenword"])[0, 0].item() == 1


def test_flow_uses_explicit_noise_and_trains_conditioning():
    head = ConditionalFlowMatchingHead(8, 6, 2, 5)
    context = torch.randn(2, 8, requires_grad=True)
    actions = torch.randn(2, 5, 2, requires_grad=True)
    targets, noise = torch.randn(2, 5, 6), torch.randn(2, 5, 6)
    loss = head.loss(context, actions, targets, noise, torch.full((2, 1), 0.4))
    loss.backward()
    assert context.grad.abs().sum() > 0 and actions.grad.abs().sum() > 0
    first = head.sample(context, actions, noise=noise, steps=4)
    torch.testing.assert_close(first, head.sample(context, actions, noise=noise, steps=4))
    assert not torch.allclose(first, head.sample(context, actions, noise=-noise, steps=4))
    with pytest.raises(ValueError):
        head.sample(context, actions, noise=noise, steps=0)


def test_end_to_end_training_reduces_finite_batch_loss():
    model, batch = MultiRateNestedWorldModel(), generate_batch(8, 10, seed=24)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    before = nested_loss(model, batch).item()
    for _ in range(12):
        optimizer.zero_grad()
        loss = nested_loss(model, batch)
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    assert nested_loss(model, batch).item() < 0.8 * before


def test_candidate_filter_enforces_constraint_independently_of_model():
    model, batch = MultiRateNestedWorldModel(), generate_batch(8, 10)
    metrics = planning_audit(model, batch)
    assert metrics["power_violation_rate"] == 0
    assert metrics["planning_regret"] >= 0 and metrics["candidate_count"] > 1


def test_persistence_negative_control_has_zero_action_effect():
    metrics = counterfactual_audit(Persistence(), generate_batch(4, 10))
    assert metrics["action_effect_nmse_db"] == pytest.approx(0)


def test_latency_reports_warmup_free_distribution():
    calls = []
    result = latency(lambda: calls.append(1), torch.device("cpu"), warmup=2, repeats=5)
    assert len(calls) == 7 and len(result["samples_ms"]) == 5
    assert 0 <= result["p50_ms"] <= result["p99_ms"]
