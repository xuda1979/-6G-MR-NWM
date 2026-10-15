"""Small finite-candidate counterfactual audit with an identical hard power constraint."""

import torch

from .data import objective_features, transition


def planning_audit(model, batch, power_cap=0.8):
    if not 0 < power_cap <= 1:
        raise ValueError("power_cap must be in (0, 1]")
    device = batch.initial.device
    levels = torch.tensor([0.1, 0.4, 0.7], device=device)
    candidates = torch.cartesian_prod(levels, levels)
    candidates = candidates[candidates.sum(-1) <= power_cap]
    size, count = batch.initial.shape[0], candidates.shape[0]
    initial = batch.initial.repeat_interleave(count, 0)
    context = batch.context.repeat_interleave(count, 0)
    intents = batch.intents.repeat_interleave(count, 0)
    actions = candidates.repeat(size, 1)
    repeated_actions = actions[:, None].repeat(1, model.config.horizon, 1)
    predicted = model(intents, initial, repeated_actions, context)
    feature_pred = objective_features(predicted["pred_states"][:, 0], actions)
    feature_true = objective_features(transition(initial, actions, context), actions)
    true_weights = batch.weights.repeat_interleave(count, 0)
    predicted_weights = predicted["objective_weights"]
    true_utility = (feature_true * true_weights).sum(-1).reshape(size, count)
    predicted_utility = (feature_pred * predicted_weights).sum(-1).reshape(size, count)
    # Oracle weights isolate physical model error from intent parsing error.
    dynamics_utility = (feature_pred * true_weights).sum(-1).reshape(size, count)
    indices = predicted_utility.argmax(-1, keepdim=True)
    dynamics_indices = dynamics_utility.argmax(-1, keepdim=True)
    best = true_utility.max(-1).values
    return {
        "planning_regret": (best - true_utility.gather(1, indices)[:, 0]).mean().item(),
        "dynamics_only_regret": (best - true_utility.gather(1, dynamics_indices)[:, 0])
        .mean()
        .item(),
        "power_violation_rate": (candidates[indices[:, 0]].sum(-1) > power_cap)
        .float()
        .mean()
        .item(),
        "candidate_count": count,
    }
