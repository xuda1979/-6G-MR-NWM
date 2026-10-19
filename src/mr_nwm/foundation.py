"""Language-model-centered MR-NWM with continuous inputs and a cached fast subnetwork.

The original small-network benchmark in model.py is intentionally unchanged.
Pretrained language weights initialize this backbone; numeric adapters and
wireless prediction heads still require task-specific training.
"""

import math
from dataclasses import dataclass, field, replace

import torch
from torch import nn

from .experts import WirelessSSMExpert
from .modalities import ModalityObservation, TypedModalityEncoder
from .model import ModelConfig, MultiRateNestedWorldModel


@dataclass
class FoundationInputs:
    intent_ids: torch.Tensor
    intent_mask: torch.Tensor
    modalities: dict[str, ModalityObservation] = field(default_factory=dict)
    evidence_ids: torch.Tensor | None = None
    evidence_mask: torch.Tensor | None = None

    def advance(self, elapsed_seconds):
        """Age a fixed observation snapshot without modifying caller-owned inputs."""
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be nonnegative and finite")
        return replace(
            self,
            modalities={
                name: replace(observation, age_seconds=observation.age_seconds + elapsed_seconds)
                for name, observation in self.modalities.items()
            },
        )


class LanguageFoundationBackbone(nn.Module):
    """Two causal views of the same pretrained language backbone.

    Physical view: numeric modalities and validated exogenous evidence only.
    Semantic view: those observations followed by operational preference text.
    The physical pass never receives the intent tokens or their hidden states.
    """

    def __init__(self, language_model, config, modality_dimensions, freeze=True, max_tokens=512):
        super().__init__()
        self.language_model = language_model
        self.freeze = freeze
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        self.max_tokens = max_tokens
        hidden = language_model.get_input_embeddings().embedding_dim
        if freeze:
            language_model.requires_grad_(False)
            language_model.eval()
        self.state = nn.Linear(config.state_dim, hidden)
        self.context = nn.Linear(config.context_dim, hidden)
        self.summary = nn.Linear(config.width, hidden)
        self.modalities = TypedModalityEncoder(modality_dimensions, hidden)
        self.physical_query = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.physical_projection = nn.Linear(hidden, config.width)
        self.semantic_projection = nn.Linear(hidden, config.width)
        self.physics = WirelessSSMExpert(config.width, config.memory)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.language_model.eval()
        return self

    def _text(self, ids, mask, batch_size):
        if ids.ndim != 2 or ids.shape[0] != batch_size or mask.shape != ids.shape:
            raise ValueError("text IDs and mask must have equal [batch, length] shape")
        if not ((mask == 0) | (mask == 1)).all() or (mask.sum(-1) == 0).any():
            raise ValueError("text mask must be binary and each text must be nonempty")
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("text IDs must be integer tensors")
        embedding = self.language_model.get_input_embeddings()
        mask = mask.to(device=embedding.weight.device, dtype=torch.bool)
        ids = ids.to(embedding.weight.device).masked_fill(~mask, 0)
        return embedding(ids), mask

    def _run(self, embeds, mask):
        if embeds.shape[1] > self.max_tokens:
            raise ValueError("slow input exceeds max_tokens; resample modalities before encoding")
        positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
        output = self.language_model(
            inputs_embeds=embeds,
            attention_mask=mask.long(),
            position_ids=positions,
            use_cache=False,
            return_dict=True,
        )
        # Freezing model weights must not disable gradients into numeric adapters.
        return output.last_hidden_state

    def forward(self, inputs, state, context, previous_summary=None):
        batch = state.shape[0]
        if previous_summary is None:
            previous_summary = state.new_zeros(batch, 1, self.summary.in_features)
        if previous_summary.shape != (batch, 1, self.summary.in_features):
            raise ValueError("previous_summary shape mismatch")
        common = torch.cat(
            [
                self.state(state)[:, None],
                self.context(context)[:, None],
                self.summary(previous_summary),
            ],
            1,
        )
        extra, extra_mask = self.modalities(inputs.modalities, batch, common)
        common = torch.cat([common, extra], 1)
        common_mask = torch.cat(
            [
                torch.ones(batch, 3, dtype=torch.bool, device=common.device),
                extra_mask,
            ],
            1,
        )
        if (inputs.evidence_ids is None) != (inputs.evidence_mask is None):
            raise ValueError("supply evidence IDs and mask together")
        if inputs.evidence_ids is not None:
            evidence, evidence_mask = self._text(inputs.evidence_ids, inputs.evidence_mask, batch)
            common = torch.cat([common, evidence], 1)
            common_mask = torch.cat([common_mask, evidence_mask], 1)
        physical_embeds = torch.cat([common, self.physical_query.expand(batch, -1, -1)], 1)
        physical_mask = torch.cat(
            [common_mask, torch.ones(batch, 1, dtype=torch.bool, device=common.device)], 1
        )
        intent, intent_mask = self._text(inputs.intent_ids, inputs.intent_mask, batch)
        if max(physical_embeds.shape[1], common.shape[1] + intent.shape[1]) > self.max_tokens:
            raise ValueError("slow input exceeds max_tokens; resample modalities before encoding")
        physical_hidden = self._run(physical_embeds, physical_mask)[:, -1]
        projected = self.physical_projection(physical_hidden)
        physical, _ = self.physics.step(projected)
        semantic_hidden = self._run(
            torch.cat([common, intent], 1), torch.cat([common_mask, intent_mask], 1)
        )[:, common.shape[1] :]
        positions = torch.arange(intent.shape[1], device=intent.device).expand(batch, -1)
        last = positions.masked_fill(~intent_mask, -1).max(-1).values
        semantic = semantic_hidden[torch.arange(batch, device=intent.device), last]
        return physical, self.semantic_projection(semantic)


class LanguageBasedWorldModel(MultiRateNestedWorldModel):
    """Use the pretrained decoder as the slow shared backbone; reuse tested fast APIs.

    Unlike the lightweight benchmark's middle-tier mixture, this model runs a
    language transformer for both physical and semantic views. The same physics
    expert is then optionally reused as the fast recurrent core. It does not
    modify the transformer's internal layers into a heterogeneous sparse MoE.
    """

    def __init__(
        self,
        language_model,
        config=ModelConfig(),
        modality_dimensions=None,
        freeze=True,
        max_tokens=512,
    ):
        nn.Module.__init__(self)
        self.config = config
        if config.use_flow:
            raise ValueError("The foundation integration does not yet train the optional flow head")
        c = config
        self.backbone = LanguageFoundationBackbone(
            language_model, c, modality_dimensions or {}, freeze, max_tokens
        )
        self.objectives = nn.Linear(c.width, 3)
        self.modulator = nn.Linear(c.width, 2 * c.width + 2 * c.width * c.rank)
        self.independent_physics = None if c.share_weights else WirelessSSMExpert(c.width, c.memory)
        self.fast_input = nn.Linear(c.state_dim + c.action_dim + c.context_dim, c.width)
        self.delta = nn.Linear(c.width, c.state_dim)
        self.resampler = nn.Conv1d(c.width, c.width, c.horizon, stride=c.horizon)
        self.flow = None
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        self.tokenizer = None
        self.pretrained_identity = None

    @classmethod
    def from_pretrained(
        cls,
        model_name,
        *,
        revision="main",
        config=ModelConfig(),
        modality_dimensions=None,
        freeze=True,
        max_tokens=512,
        dtype=torch.float32,
    ):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "Install the optional foundation dependencies: pip install -e '.[foundation]'"
            ) from error
        language_model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=False,
            torch_dtype=dtype,
            attn_implementation="eager",
            use_safetensors=True,
        )
        model = cls(language_model, config, modality_dimensions, freeze, max_tokens).to(dtype=dtype)
        model.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, trust_remote_code=False
        )
        if model.tokenizer.pad_token_id is None:
            model.tokenizer.pad_token = model.tokenizer.eos_token
        model.pretrained_identity = {
            "model": model_name,
            "requested_revision": revision,
            "resolved_revision": getattr(language_model.config, "_commit_hash", None),
        }
        return model

    def tokenize(self, intents, *, modalities=None, evidence=None):
        if self.tokenizer is None:
            raise ValueError("tokenize requires a tokenizer loaded by from_pretrained")
        device = self.delta.weight.device
        encoded = self.tokenizer(intents, padding=True, return_tensors="pt", truncation=False)
        inputs = FoundationInputs(
            encoded.input_ids.to(device), encoded.attention_mask.to(device), modalities or {}
        )
        if evidence is not None:
            extra = self.tokenizer(evidence, padding=True, return_tensors="pt", truncation=False)
            inputs.evidence_ids, inputs.evidence_mask = (
                extra.input_ids.to(device),
                extra.attention_mask.to(device),
            )
        return inputs

    def rollout(self, intents, initial_state, actions, context, *, macro_period_seconds=None):
        """Forecast with a fixed observation snapshot that ages at each macro boundary.

        The physical clock must be supplied for multi-window multimodal forecasts.
        For new measurements or context, call forward per window with sync_summary.
        """
        horizon = self.config.horizon
        if actions.ndim != 3 or actions.shape[1] == 0 or actions.shape[1] % horizon:
            raise ValueError("rollout length must be a positive multiple of horizon")
        if macro_period_seconds is not None and (
            not math.isfinite(macro_period_seconds) or macro_period_seconds <= 0
        ):
            raise ValueError("macro_period_seconds must be positive and finite")
        if intents.modalities and actions.shape[1] > horizon and macro_period_seconds is None:
            raise ValueError("multi-window modalities require macro_period_seconds")
        state, summary, chunks = initial_state, None, []
        for index, window in enumerate(actions.split(horizon, dim=1)):
            inputs = intents.advance(index * (macro_period_seconds or 0))
            output = self(inputs, state, window, context, summary)
            chunks.append(output["pred_states"])
            state, summary = chunks[-1][:, -1], output["sync_summary"]
        return torch.cat(chunks, 1)

    def forward_slow_tier(self, intent_tokens, state, context, previous_summary=None):
        c = self.config
        if state.ndim != 2 or state.shape[-1] != c.state_dim:
            raise ValueError("state shape mismatch")
        if context.shape != (state.shape[0], c.context_dim):
            raise ValueError("context shape mismatch")
        physical, semantic = self.backbone(intent_tokens, state, context, previous_summary)
        if c.detach_bridge:
            physical = physical.detach()
        raw = self.modulator(physical)
        gamma, beta, a, b = torch.split(
            raw, [c.width, c.width, c.width * c.rank, c.width * c.rank], -1
        )
        if c.modulation:
            gamma, beta, a, b = (
                1 + 0.1 * gamma.tanh(),
                0.1 * beta.tanh(),
                0.1 * a.tanh(),
                0.1 * b.tanh(),
            )
        else:
            gamma, beta, a, b = torch.ones_like(gamma), 0 * beta, 0 * a, 0 * b
        return {
            "gamma": gamma,
            "beta": beta,
            "lora_a": a.reshape(-1, c.width, c.rank),
            "lora_b": b.reshape(-1, c.width, c.rank),
            "context": context,
            "physical_context": physical,
            "objective_weights": self.objectives(semantic).softmax(-1),
            "alignment_loss": state.new_zeros(()),
        }
