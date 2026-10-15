"""Optional action-conditioned flow-matching head with an explicit base noise."""

import torch
from torch import nn
from torch.nn import functional as F


class ConditionalFlowMatchingHead(nn.Module):
    def __init__(self, width, state_dim, action_dim, horizon):
        super().__init__()
        self.horizon, self.state_dim = horizon, state_dim
        self.size = horizon * state_dim
        inputs = self.size + horizon * action_dim + width + 1
        self.velocity = nn.Sequential(
            nn.Linear(inputs, 2 * width), nn.SiLU(), nn.Linear(2 * width, self.size)
        )

    def loss(self, context, actions, targets, noise=None, time=None):
        target = targets.flatten(1)
        noise = torch.randn_like(target) if noise is None else noise.flatten(1)
        if time is None:
            time = torch.rand(target.shape[0], 1, device=target.device, dtype=target.dtype)
        point = (1 - time) * noise + time * target
        velocity = self.velocity(torch.cat([point, actions.flatten(1), context, time], -1))
        return F.mse_loss(velocity, target - noise)

    def sample(self, context, actions, *, noise, steps=8):
        """Integrate from the caller's Gaussian draw; zero is only a diagnostic path."""
        if steps < 1:
            raise ValueError("steps must be positive")
        if noise.shape != (context.shape[0], self.horizon, self.state_dim):
            raise ValueError("noise must have shape [batch, horizon, state_dim]")
        point = noise.flatten(1)
        for index in range(steps):
            time = point.new_full((point.shape[0], 1), index / steps)
            point = (
                point
                + self.velocity(torch.cat([point, actions.flatten(1), context, time], -1)) / steps
            )
        return point.reshape(-1, self.horizon, self.state_dim)
