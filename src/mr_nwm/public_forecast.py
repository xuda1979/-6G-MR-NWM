"""Observed-trace forecasting with train-only transforms and frozen test membership.

No unobserved power/PRB actions or intent labels are manufactured. The pretrained
language trunk is frozen; continuous adapters, modulation and dynamics are trained.
"""

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .foundation import FoundationInputs, LanguageBasedWorldModel
from .modalities import ModalityObservation
from .model import ModelConfig, MultiRateNestedWorldModel
from .run_store import save_checkpoint

LANGUAGE_MODEL = "HuggingFaceTB/SmolLM2-135M"
LANGUAGE_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
BACKBONE_PREFIX = "backbone.language_model."


def contract_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass
class ForecastScaler:
    mean: list
    scale: list

    @classmethod
    def fit(cls, split):
        values = np.asarray(split.history, dtype=np.float64).copy()
        valid = np.asarray(split.observed, dtype=bool)
        if not np.isfinite(values[valid]).all() or (values[..., 0][valid[..., 0]] < 0).any():
            raise ValueError("training observations must be finite and target nonnegative")
        values[..., 0] = np.log1p(np.maximum(values[..., 0], 0))
        means, scales = [], []
        for feature in range(values.shape[-1]):
            selected = values[..., feature][valid[..., feature]]
            means.append(float(selected.mean()) if selected.size else 0.0)
            scales.append(max(float(selected.std()), 0.1) if selected.size else 1.0)
        return cls(means, scales)

    def transform(self, split):
        values = np.asarray(split.history, dtype=np.float64).copy()
        valid = np.asarray(split.observed, dtype=bool)
        if not valid[:, -1, 0].all():
            raise ValueError("last observed target is required for persistence")
        values[..., 0] = np.log1p(np.maximum(values[..., 0], 0))
        values = np.where(valid, (values - self.mean) / self.scale, 0).astype(np.float32)
        telemetry = np.concatenate([values, valid.astype(np.float32)], -1)
        periods = np.asarray(getattr(split, "periods", np.ones(len(values))), dtype=np.float32)
        if not np.isfinite(periods).all() or (periods <= 0).any():
            raise ValueError("periods must be positive and finite")
        targets = np.asarray(split.targets, dtype=np.float64)
        if not np.isfinite(targets).all() or (targets < 0).any():
            raise ValueError("forecast targets must be finite and nonnegative")
        return {
            "telemetry": telemetry,
            "context": np.concatenate(
                [telemetry.reshape(len(values), -1), np.log1p(periods[:, None])], -1
            ).astype(np.float32),
            "initial": values[:, -1, :1],
            "targets": ((np.log1p(targets) - self.mean[0]) / self.scale[0]).astype(np.float32),
            "physical_targets": targets,
            "periods": periods,
        }

    def inverse(self, prediction):
        log_values = np.asarray(prediction, dtype=np.float64) * self.scale[0] + self.mean[0]
        if not np.isfinite(log_values).all():
            raise ValueError("non-finite model output")
        # Nonnegative physical quantities; cap only to prevent exponential overflow.
        return np.expm1(np.clip(log_values, 0, 30))

    def to_dict(self):
        return asdict(self)


def select_indices(sample_ids, limit):
    if limit <= 0 or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("positive evaluation limit and unique sample IDs required")
    return np.array(
        sorted(
            range(len(sample_ids)), key=lambda i: hashlib.sha256(sample_ids[i].encode()).digest()
        )[:limit],
        dtype=np.int64,
    )


def physical_metrics(prediction, targets):
    prediction, targets = np.asarray(prediction), np.asarray(targets)
    if prediction.shape != targets.shape or not prediction.size:
        raise ValueError("nonempty prediction and target shapes must match")
    if not np.isfinite(prediction).all() or not np.isfinite(targets).all():
        raise ValueError("metrics require finite predictions and targets")
    error = prediction.astype(np.float64) - targets
    energy = max(float(np.square(targets).sum()), 1e-12)
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "nmse_db": float(10 * np.log10(max(float(np.square(error).sum()) / energy, 1e-12))),
        "horizon_mae": np.abs(error).mean(0).tolist(),
    }


def adapter_state(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if not key.startswith(BACKBONE_PREFIX)
    }


def frozen_backbone_hash(model):
    if not isinstance(model, LanguageBasedWorldModel):
        return None
    digest = hashlib.sha256()
    for name, parameter in model.backbone.language_model.named_parameters():
        if parameter.requires_grad:
            raise ValueError("public forecast contract requires a frozen language trunk")
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def restore_adapters(model, state):
    expected = {key for key in model.state_dict() if not key.startswith(BACKBONE_PREFIX)}
    if set(state) != expected:
        raise ValueError("incomplete or incompatible adapter checkpoint")
    model.load_state_dict(state, strict=False)


def build_forecaster(kind, history=8, horizon=5, *, language_model=None):
    config = ModelConfig(
        state_dim=1,
        action_dim=1,
        context_dim=history * 8 + 1,
        horizon=horizon,
        truncation=0,
    )
    if kind == "foundation":
        if language_model is None:
            model = LanguageBasedWorldModel.from_pretrained(
                LANGUAGE_MODEL,
                revision=LANGUAGE_REVISION,
                config=config,
                modality_dimensions={"telemetry": 8},
                freeze=True,
            )
        else:
            model = LanguageBasedWorldModel(
                language_model, config, modality_dimensions={"telemetry": 8}, freeze=True
            )
        # This observational task supplies no preference/objective supervision.
        model.backbone.semantic_projection.requires_grad_(False)
        model.intent_ids = torch.tensor([[1]], dtype=torch.long)
        if model.tokenizer is not None:
            model.intent_ids = model.tokenizer(
                "Predict network measurements.", return_tensors="pt"
            )["input_ids"]
    elif kind == "numeric":
        model = MultiRateNestedWorldModel(config)
    else:
        raise ValueError(f"unknown forecast model: {kind}")
    model.objectives.requires_grad_(False)
    return model


def predict_tensor(model, encoded, indices, nominal_period):
    initial = torch.from_numpy(encoded["initial"][indices])
    context = torch.from_numpy(encoded["context"][indices])
    actions = torch.zeros(len(indices), model.config.horizon, 1)
    if isinstance(model, LanguageBasedWorldModel):
        values = torch.from_numpy(encoded["telemetry"][indices])
        periods = torch.from_numpy(encoded["periods"][indices])
        ages = torch.arange(values.shape[1] - 1, -1, -1).float()[None] * periods[:, None]
        intents = model.intent_ids.expand(len(indices), -1)
        inputs = FoundationInputs(
            intent_ids=intents,
            intent_mask=torch.ones_like(intents, dtype=torch.bool),
            modalities={
                "telemetry": ModalityObservation(values, ages, nominal_period),
            },
        )
    else:
        inputs = torch.ones(len(indices), 1, dtype=torch.long)
    return model(inputs, initial, actions, context)["pred_states"][..., 0]


@torch.inference_mode()
def predict_numpy(model, encoded, indices, nominal_period, batch_size=32):
    model.eval()
    return np.concatenate(
        [
            predict_tensor(model, encoded, chunk, nominal_period).numpy()
            for chunk in np.array_split(indices, math.ceil(len(indices) / batch_size))
        ]
    )


def ridge_forecast(train, evaluation, alpha=10.0):
    x = np.column_stack([train["context"], np.ones(len(train["context"]))]).astype(np.float64)
    y = train["targets"] - train["initial"]
    regularizer = np.eye(x.shape[1]) * alpha
    regularizer[-1, -1] = 0
    coefficients = np.linalg.solve(x.T @ x + regularizer, x.T @ y)
    features = np.column_stack([evaluation["context"], np.ones(len(evaluation["context"]))])
    return evaluation["initial"] + features @ coefficients, coefficients


def train_forecaster(
    model, train, validation, val_indices, scaler, config, run_dir, identity, *, stop_after=None
):
    """Validation selects the checkpoint; test targets never enter this function."""
    run_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["learning_rate"],
        weight_decay=1e-4,
    )
    rng = np.random.default_rng(config["seed"])
    start_step, best_score, best_step = 0, math.inf, 0
    best_state, trace, seen = None, [], set()
    resume_file = run_dir / "resume.pt"
    if resume_file.exists():
        saved = torch.load(resume_file, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("resume identity mismatch")
        restore_adapters(model, saved["adapters"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = saved["sampler"]
        torch.set_rng_state(saved["torch_rng"])
        start_step, best_score = saved["step"], saved["best_score"]
        best_step, best_state, trace = saved["best_step"], saved["best_state"], saved["trace"]
        seen = set(saved["seen"])
    initial_state = adapter_state(model)
    frozen_before = frozen_backbone_hash(model)
    start = time.perf_counter()
    nominal_period = float(np.median(train["periods"]))
    final_step = min(config["updates"], stop_after or config["updates"])
    for step in range(start_step, final_step):
        model.train()
        indices = rng.integers(0, len(train["initial"]), size=config["batch_size"])
        seen.update(map(int, indices))
        optimizer.zero_grad(set_to_none=True)
        prediction = predict_tensor(model, train, indices, nominal_period)
        target = torch.from_numpy(train["targets"][indices])
        loss = torch.nn.functional.mse_loss(prediction, target)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % config["validation_every"] == 0 or step + 1 == config["updates"]:
            forecast = scaler.inverse(predict_numpy(model, validation, val_indices, nominal_period))
            truth = validation["physical_targets"][val_indices]
            score = physical_metrics(forecast, truth)["mae"]
            trace.append(
                {
                    "step": step + 1,
                    "train_loss": loss.item(),
                    "gradient_norm": gradient.item(),
                    "validation_mae": score,
                }
            )
            if score < best_score:
                best_score, best_step, best_state = score, step + 1, adapter_state(model)
            save_checkpoint(
                resume_file,
                {
                    "identity": identity,
                    "adapters": adapter_state(model),
                    "optimizer": optimizer.state_dict(),
                    "sampler": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "step": step + 1,
                    "best_score": best_score,
                    "best_step": best_step,
                    "best_state": best_state,
                    "trace": trace,
                    "seen": sorted(seen),
                },
            )
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "validation_mae": score,
                        "best_step": best_step,
                        "seconds": time.perf_counter() - start,
                    }
                ),
                flush=True,
            )
    if best_state is None:
        raise ValueError("no finite validation checkpoint")
    restore_adapters(model, best_state)
    frozen_after = frozen_backbone_hash(model)
    if frozen_before != frozen_after:
        raise RuntimeError("frozen language backbone changed during training")
    components = {}
    for prefix in (
        "backbone.state.",
        "backbone.context.",
        "backbone.modalities.adapters.",
        "backbone.physical_projection.",
        "modulator.",
        "fast_input.",
        "delta.",
    ):
        differences = [
            (best_state[key] - initial_state[key]).abs().max().item()
            for key in best_state
            if key.startswith(prefix)
        ]
        if differences:
            components[prefix] = max(differences)
    update = max((best_state[key] - initial_state[key]).abs().max().item() for key in best_state)
    if start_step == 0 and update == 0:
        raise RuntimeError("training did not change model parameters")
    return {
        "selected_step": best_step,
        "validation_mae": best_score,
        "training_seconds_this_invocation": time.perf_counter() - start,
        "trace": trace,
        "unique_training_windows_seen": len(seen),
        "max_parameter_change_this_invocation": update,
        "selected_component_parameter_changes_this_invocation": components,
        "frozen_backbone_sha256": frozen_after,
        "nominal_period_seconds": nominal_period,
    }
