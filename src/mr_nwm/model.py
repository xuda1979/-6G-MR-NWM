"""Nested multi-rate model with a causal separation of intent and dynamics."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .experts import HeterogeneousMoEBackbone, WirelessSSMExpert
from .flow import ConditionalFlowMatchingHead


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32
    state_dim: int = 6
    action_dim: int = 2
    context_dim: int = 4
    width: int = 32
    memory: int = 16
    heads: int = 4
    rank: int = 2
    horizon: int = 5
    truncation: int = 3
    share_weights: bool = True
    modulation: bool = True
    detach_bridge: bool = False
    use_flow: bool = False

    def __post_init__(self):
        for name in (
            "vocab_size",
            "state_dim",
            "action_dim",
            "context_dim",
            "width",
            "memory",
            "heads",
            "rank",
            "horizon",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.width % self.heads or self.truncation < 0:
            raise ValueError("width must divide into heads; truncation must be nonnegative")


class MultiRateNestedWorldModel(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config
        c = config
        self.embedding = nn.Embedding(c.vocab_size, c.width, padding_idx=0)
        self.position = nn.Parameter(torch.randn(1, 32, c.width) * 0.01)
        self.state_encoder = nn.Linear(c.state_dim, c.width)
        self.context_encoder = nn.Linear(c.context_dim, c.width)
        self.backbone = HeterogeneousMoEBackbone(c.width, c.memory, c.heads)
        self.objectives = nn.Linear(c.width, 3)
        self.modulator = nn.Linear(c.width, 2 * c.width + 2 * c.width * c.rank)
        if c.share_weights:
            # A property aliases this module; avoid duplicate state_dict registrations.
            self.independent_physics = None
        else:
            self.independent_physics = WirelessSSMExpert(c.width, c.memory)
        self.fast_input = nn.Linear(c.state_dim + c.action_dim + c.context_dim, c.width)
        self.delta = nn.Linear(c.width, c.state_dim)
        self.resampler = nn.Conv1d(c.width, c.width, c.horizon, stride=c.horizon)
        self.flow = (
            ConditionalFlowMatchingHead(c.width, c.state_dim, c.action_dim, c.horizon)
            if c.use_flow
            else None
        )
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    @property
    def fast_physics_core(self):
        if self.config.share_weights:
            return self.backbone.physics
        return self.independent_physics

    def forward_slow_tier(self, intent_tokens, state, context, previous_summary=None):
        c = self.config
        if intent_tokens.ndim != 2 or not 0 < intent_tokens.shape[1] <= 32:
            raise ValueError("intent_tokens must have shape [batch, 1..32]")
        if (intent_tokens != 0).sum(1).min() == 0:
            raise ValueError("each intent must contain a non-padding token")
        if state.shape != (intent_tokens.shape[0], c.state_dim):
            raise ValueError("state shape mismatch")
        if context.shape != (state.shape[0], c.context_dim):
            raise ValueError("context shape mismatch")
        language = self.embedding(intent_tokens) + self.position[:, : intent_tokens.shape[1]]
        if previous_summary is None:
            previous_summary = state.new_zeros(state.shape[0], 1, c.width)
        if previous_summary.shape != (state.shape[0], 1, c.width):
            raise ValueError("previous_summary shape mismatch")
        physical = torch.cat(
            [
                self.state_encoder(state)[:, None],
                self.context_encoder(context)[:, None],
                previous_summary,
            ],
            dim=1,
        )
        padding = torch.cat(
            [
                intent_tokens == 0,
                torch.zeros(state.shape[0], 3, device=state.device, dtype=torch.bool),
            ],
            dim=1,
        )
        hidden, routing, alignment = self.backbone(
            torch.cat([language, physical], 1), intent_tokens.shape[1], padding
        )
        semantic = (hidden[:, : language.shape[1]] * (intent_tokens != 0)[..., None]).sum(1)
        semantic = semantic / (intent_tokens != 0).sum(1, keepdim=True)
        physical_context = hidden[:, language.shape[1] :].mean(1)
        if c.detach_bridge:
            physical_context = physical_context.detach()
        raw = self.modulator(physical_context)
        gamma, beta, a, b = torch.split(
            raw, [c.width, c.width, c.width * c.rank, c.width * c.rank], dim=-1
        )
        gamma, beta = 1 + 0.1 * gamma.tanh(), 0.1 * beta.tanh()
        a, b = 0.1 * a.tanh(), 0.1 * b.tanh()
        if not c.modulation:
            gamma, beta, a, b = torch.ones_like(gamma), 0 * beta, 0 * a, 0 * b
        return {
            "gamma": gamma,
            "beta": beta,
            "lora_a": a.reshape(-1, c.width, c.rank),
            "lora_b": b.reshape(-1, c.width, c.rank),
            "objective_weights": self.objectives(semantic).softmax(-1),
            "context": context,
            "physical_context": physical_context,
            "routing_weights": routing,
            "alignment_loss": alignment,
        }

    def forward_fast_tier_step(self, state, action, packet, memory=None):
        hidden = F.silu(self.fast_input(torch.cat([state, action, packet["context"]], -1)))
        hidden = packet["gamma"] * hidden + packet["beta"]
        hidden, memory = self.fast_physics_core.step(
            hidden, memory, packet["lora_a"], packet["lora_b"]
        )
        return state + self.delta(hidden), hidden, memory

    def forward(self, intents, initial_state, actions, context, previous_summary=None):
        c = self.config
        if actions.shape != (initial_state.shape[0], c.horizon, c.action_dim):
            raise ValueError("actions must have shape [batch, horizon, action_dim]")
        packet = self.forward_slow_tier(intents, initial_state, context, previous_summary)
        state, memory, predictions, hiddens = initial_state, None, [], []
        for step in range(c.horizon):
            if step and c.truncation and step % c.truncation == 0:
                state, memory = state.detach(), memory.detach()
            state, hidden, memory = self.forward_fast_tier_step(
                state, actions[:, step], packet, memory
            )
            predictions.append(state)
            hiddens.append(hidden)
        summary = self.resampler(torch.stack(hiddens, -1)).transpose(1, 2)
        return {"pred_states": torch.stack(predictions, 1), "sync_summary": summary, **packet}

    def rollout(self, intents, initial_state, actions, context):
        """Free-running macro windows: advance state and feedback; never reset to s0."""
        if actions.ndim != 3 or actions.shape[1] % self.config.horizon:
            raise ValueError("rollout length must be a positive multiple of horizon")
        if actions.shape[1] == 0:
            raise ValueError("rollout cannot be empty")
        state, summary, chunks = initial_state, None, []
        for window in actions.split(self.config.horizon, dim=1):
            output = self(intents, state, window, context, summary)
            chunks.append(output["pred_states"])
            state, summary = chunks[-1][:, -1], output["sync_summary"]
        return torch.cat(chunks, 1)
