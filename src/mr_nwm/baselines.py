"""Actual executable baselines; no unimplemented language-model comparison."""

import torch
from torch import nn

from .experts import WirelessSSMExpert


class Persistence(nn.Module):
    def rollout(self, intents, initial, actions, context):
        return initial[:, None].expand(-1, actions.shape[1], -1)


class ResidualMLP(nn.Module):
    """Action/context-conditioned autoregressive predictor trained on the same data."""

    def __init__(self, width=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(12, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 6),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def rollout(self, intents, initial, actions, context):
        state, predictions = initial, []
        for action in actions.unbind(1):
            state = state + self.network(torch.cat([state, action, context], -1))
            predictions.append(state)
        return torch.stack(predictions, 1)


class StandaloneSSM(nn.Module):
    """Same fast operator and truncation schedule, without a slow tier or modulation."""

    def __init__(self, horizon=5, truncation=3, width=32, memory=16):
        super().__init__()
        self.horizon, self.truncation = horizon, truncation
        self.input = nn.Linear(12, width)
        self.physics = WirelessSSMExpert(width, memory)
        self.delta = nn.Linear(width, 6)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def rollout(self, intents, initial, actions, context):
        state, memory, predictions = initial, None, []
        for index, action in enumerate(actions.unbind(1)):
            step = index % self.horizon
            if step == 0:
                memory = None
            elif self.truncation and step % self.truncation == 0:
                state, memory = state.detach(), memory.detach()
            hidden = torch.nn.functional.silu(self.input(torch.cat([state, action, context], -1)))
            hidden, memory = self.physics.step(hidden, memory)
            state = state + self.delta(hidden)
            predictions.append(state)
        return torch.stack(predictions, 1)
