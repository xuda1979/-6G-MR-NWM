"""Continuous modality tokens with validity, observation age, and optional graph edges."""

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModalityObservation:
    values: torch.Tensor  # [batch, tokens, features]; CSI/IQ use real/imag feature pairs
    age_seconds: torch.Tensor  # [batch, tokens], relative to the current slow update
    period_seconds: float
    valid: torch.Tensor | None = None  # [batch, tokens], True means observed
    adjacency: torch.Tensor | None = None  # optional nonnegative [batch, nodes, nodes]
    max_age_seconds: float | None = None  # optional expiry; acquisition period is not a TTL


class TypedModalityEncoder(nn.Module):
    """Trainable continuous front ends, not pretrained CSI/vision foundation encoders.

    Graph inputs optionally get one normalized message-passing step. Images must
    already be patched; complex CSI and IQ must be represented by real/imag pairs.
    This adapter preserves numeric inputs instead of serializing them into text.
    """

    def __init__(self, dimensions, hidden_size):
        super().__init__()
        if any(not isinstance(size, int) or size <= 0 for size in dimensions.values()):
            raise ValueError("modality feature dimensions must be positive integers")
        self.names = tuple(sorted(dimensions))
        self.adapters = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Identity(),
                    nn.Linear(dimensions[name], hidden_size),
                    nn.GELU(),
                )
                for name in self.names
            }
        )
        self.type_embedding = nn.Embedding(max(1, len(self.names)), hidden_size)
        self.time_embedding = nn.Linear(2, hidden_size)
        self.graph_message = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, observations, batch_size, reference):
        unknown = set(observations) - set(self.names)
        if unknown:
            raise ValueError(f"unregistered modalities: {sorted(unknown)}")
        tokens, masks = [], []
        for index, name in enumerate(self.names):
            if name not in observations:
                continue
            observation = observations[name]
            values = observation.values
            if values.ndim != 3 or values.shape[0] != batch_size or values.shape[1] == 0:
                raise ValueError(f"{name}: expected [batch, nonempty tokens, features]")
            expected_features = self.adapters[name][1].in_features
            if values.shape[-1] != expected_features or values.is_complex():
                raise ValueError(f"{name}: supply {expected_features} real-valued features")
            if observation.age_seconds.shape != values.shape[:2]:
                raise ValueError(f"{name}: age shape mismatch")
            valid = observation.valid
            if valid is None:
                valid = torch.ones(values.shape[:2], device=values.device, dtype=torch.bool)
            if valid.shape != values.shape[:2] or valid.dtype != torch.bool:
                raise ValueError(f"{name}: valid must be a boolean [batch, tokens] mask")
            ages = observation.age_seconds
            if ages.is_complex():
                raise ValueError(f"{name}: observation ages must be real-valued")
            period = observation.period_seconds
            if not math.isfinite(period) or period <= 0:
                raise ValueError(f"{name}: period must be positive and finite")
            if not torch.isfinite(ages[valid]).all():
                raise ValueError(f"{name}: valid observation ages must be finite")
            if (ages[valid] < 0).any():
                raise ValueError(f"{name}: future observations are not allowed")
            max_age = observation.max_age_seconds
            if max_age is not None:
                if not math.isfinite(max_age) or max_age < 0:
                    raise ValueError(f"{name}: max_age_seconds must be nonnegative and finite")
                valid = valid & (ages <= max_age)
            if not torch.isfinite(values[valid]).all():
                raise ValueError(f"{name}: valid observations must be finite")
            valid = valid.to(reference.device)
            values = values.to(device=reference.device, dtype=reference.dtype)
            values = torch.where(valid[..., None], values, 0)
            if not torch.isfinite(values).all():
                raise ValueError(f"{name}: valid observations must be representable in model dtype")
            # Compress seconds before narrowing: a finite age can overflow FP16
            # even though its logarithm is small. Preserve FP64 source ranges too.
            ages = ages.to(
                device=reference.device, dtype=torch.promote_types(ages.dtype, torch.float32)
            )
            ages = torch.where(valid, ages, 0)
            hidden = self.adapters[name](values)
            if observation.adjacency is not None:
                adjacency = observation.adjacency
                count = values.shape[1]
                if adjacency.shape != (batch_size, count, count):
                    raise ValueError(f"{name}: adjacency shape mismatch")
                if adjacency.is_complex():
                    raise ValueError(f"{name}: adjacency must be real-valued")
                adjacency = adjacency.to(
                    device=reference.device,
                    dtype=torch.promote_types(adjacency.dtype, torch.float32),
                )
                if not torch.isfinite(adjacency).all() or (adjacency < 0).any():
                    raise ValueError(f"{name}: adjacency must be finite and nonnegative")
                adjacency = adjacency * valid[:, :, None] * valid[:, None, :]
                # Normalize before narrowing so small weights do not disappear and
                # large weights do not overflow. Zero rows remain isolated.
                scale = adjacency.amax(-1, keepdim=True)
                adjacency = adjacency / torch.where(scale > 0, scale, 1)
                degree = adjacency.sum(-1, keepdim=True)
                adjacency = adjacency / torch.where(degree > 0, degree, 1)
                adjacency = adjacency.to(dtype=reference.dtype)
                hidden = hidden + self.graph_message(torch.bmm(adjacency, hidden))
            time_features = torch.stack(
                [torch.log1p(ages), torch.full_like(ages, math.log1p(period))], -1
            ).to(dtype=reference.dtype)
            hidden = hidden + self.time_embedding(time_features) + self.type_embedding.weight[index]
            tokens.append(torch.where(valid[..., None], hidden, 0))
            masks.append(valid)
        if not tokens:
            return reference.new_zeros(batch_size, 0, reference.shape[-1]), torch.zeros(
                batch_size, 0, dtype=torch.bool, device=reference.device
            )
        return torch.cat(tokens, 1), torch.cat(masks, 1)
