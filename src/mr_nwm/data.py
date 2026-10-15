"""Two-link normalized queue/interference simulator; not a 3GPP channel model."""

from dataclasses import dataclass

import torch

from .intent import intent_batch

STATE_NAMES = ("gain_1", "gain_2", "queue_1", "queue_2", "interference_1", "interference_2")
CONTEXT_NAMES = ("arrival_1", "arrival_2", "cross_gain", "mobility")


def transition(state, action, context, noise=None):
    """Causal transition. Preference/intent is deliberately absent from the physics."""
    gain, queue, interference = state.split(2, dim=-1)
    arrival, cross, mobility = context[:, :2], context[:, 2:3], context[:, 3:4]
    noise = torch.zeros_like(gain) if noise is None else noise
    next_gain = ((1 - mobility) * gain + mobility * 0.55 + noise).clamp(0.05, 1.2)
    rate = torch.log2(1 + 4 * gain * action / (0.2 + interference + cross * action.flip(-1)))
    next_queue = (queue + arrival - 0.12 * rate).clamp(0, 2)
    next_interference = 0.65 * interference + 0.35 * cross * action.flip(-1)
    return torch.cat([next_gain, next_queue, next_interference], -1)


def objective_features(states, actions):
    gain, queue, interference = states.split(2, dim=-1)
    # Post-transition throughput proxy, normalized identically for all planners.
    throughput = torch.log2(1 + 4 * gain * actions / (0.2 + interference)).mean(-1) / 3
    return torch.stack([throughput, -queue.mean(-1), -actions.mean(-1)], -1)


@dataclass
class Batch:
    intents: torch.Tensor
    weights: torch.Tensor
    initial: torch.Tensor
    actions: torch.Tensor
    context: torch.Tensor
    targets: torch.Tensor

    def to(self, device):
        return Batch(**{name: value.to(device) for name, value in vars(self).items()})


def generate_batch(size=32, steps=10, seed=0, scenario="iid", intent_split="train"):
    if size <= 0 or steps <= 0:
        raise ValueError("size and steps must be positive")
    if scenario not in {"iid", "ood"}:
        raise ValueError("scenario must be iid or ood")
    generator = torch.Generator().manual_seed(seed)
    intents, weights = intent_batch(size, generator, intent_split)
    initial = torch.cat(
        [
            0.2 + 0.7 * torch.rand(size, 2, generator=generator),
            0.1 + 0.8 * torch.rand(size, 2, generator=generator),
            0.05 + 0.2 * torch.rand(size, 2, generator=generator),
        ],
        -1,
    )
    arrivals = 0.05 + 0.1 * torch.rand(size, 2, generator=generator)
    cross = 0.08 + 0.17 * torch.rand(size, 1, generator=generator)
    mobility = 0.03 + 0.09 * torch.rand(size, 1, generator=generator)
    if scenario == "ood":
        arrivals = arrivals + 0.12
        cross = cross + 0.3
        mobility = mobility + 0.25
    context = torch.cat([arrivals, cross, mobility], -1)
    actions = torch.rand(size, steps, 2, generator=generator)
    state, targets = initial, []
    for step in range(steps):
        noise = 0.005 * torch.randn(size, 2, generator=generator)
        state = transition(state, actions[:, step], context, noise)
        targets.append(state)
    return Batch(intents, weights, initial, actions, context, torch.stack(targets, 1))
