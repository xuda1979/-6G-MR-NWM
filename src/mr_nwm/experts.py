"""Dense heterogeneous expert mixture and explicit recurrent physics memory."""

import torch
from torch import nn
from torch.nn import functional as F


class WirelessSSMExpert(nn.Module):
    """Input-dependent gated recurrence, inspired by SSMs; not a Mamba implementation."""

    def __init__(self, width: int, memory: int):
        super().__init__()
        self.memory = memory
        self.norm = nn.LayerNorm(width)
        self.input = nn.Linear(width, 2 * width)
        self.decay = nn.Linear(width, memory)
        self.write = nn.Linear(width, memory, bias=False)
        self.read = nn.Linear(memory, width, bias=False)
        self.output = nn.Linear(width, width)

    def step(self, x, state=None, lora_a=None, lora_b=None):
        if state is None:
            state = x.new_zeros(x.shape[0], self.memory)
        normalized = self.norm(x)
        u, gate = self.input(normalized).chunk(2, dim=-1)
        u = F.silu(u)
        decay = torch.sigmoid(self.decay(u))
        state = decay * state + (1 - decay) * torch.tanh(self.write(u))
        output = self.output(self.read(state) * torch.sigmoid(gate))
        if (lora_a is None) != (lora_b is None):
            raise ValueError("Supply both low-rank factors or neither")
        if lora_a is not None:
            # Row-vector convention: x A B^T / sqrt(rank). Never materialize D x D.
            output = (
                output
                + torch.bmm(torch.bmm(normalized[:, None], lora_a), lora_b.transpose(1, 2))[:, 0]
                / lora_a.shape[-1] ** 0.5
            )
        return x + output, state

    def forward(self, sequence):
        memory, outputs = None, []
        for token in sequence.unbind(1):
            output, memory = self.step(token, memory)
            outputs.append(output)
        return torch.stack(outputs, dim=1)


class HeterogeneousMoEBackbone(nn.Module):
    """All experts execute: this dense mixture does not claim sparse-MoE savings.

    Physical queries cannot attend to intent tokens. Physics recurrence starts at
    the first physical token, preventing an indirect semantic-to-physics leak.
    """

    def __init__(self, width=32, memory=16, heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, dropout=0, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.semantic = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )
        self.physics = WirelessSSMExpert(width, memory)
        self.router = nn.Linear(width + 2, 2)

    def forward(self, sequence, language_length, padding):
        length = sequence.shape[1]
        mask = torch.zeros(length, length, device=sequence.device, dtype=torch.bool)
        mask[language_length:, :language_length] = True
        attended, _ = self.attention(
            sequence,
            sequence,
            sequence,
            attn_mask=mask,
            key_padding_mask=padding,
            need_weights=False,
        )
        hidden = self.norm(sequence + attended)
        modality = torch.ones_like(padding, dtype=torch.long)
        modality[:, :language_length] = 0
        features = F.one_hot(modality, 2).to(hidden.dtype)
        weights = self.router(torch.cat([hidden, features], dim=-1)).softmax(-1)
        semantic = hidden + self.semantic(hidden)
        physical = torch.cat(
            [hidden[:, :language_length], self.physics(hidden[:, language_length:])], dim=1
        )
        mixed = weights[..., :1] * semantic + weights[..., 1:] * physical
        assigned = weights.gather(-1, modality.unsqueeze(-1)).squeeze(-1)
        alignment = -assigned[~padding].clamp_min(1e-8).log().mean()
        return mixed, weights, alignment
