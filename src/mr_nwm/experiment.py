"""Fixed-budget multi-seed training and held-out evaluation, without test-set selection."""

import argparse
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock, Timeout
from torch.nn import functional as F

from .baselines import Persistence, ResidualMLP, StandaloneSSM
from .data import generate_batch, transition
from .metrics import intent_metrics, latency, nmse_db
from .model import ModelConfig, MultiRateNestedWorldModel
from .planning import planning_audit
from .provenance import sha256_file, source_hashes
from .run_store import MANIFEST_VERSION, load_resume_manifest, save_checkpoint, save_json
from .verification import compare_metrics, load_verified_checkpoint

VARIANTS = (
    "full",
    "unshared",
    "no_modulation",
    "detached",
    "mlp",
    "mlp_matched",
    "ssm",
    "persistence",
)


def build_model(name, horizon):
    if name not in VARIANTS:
        raise ValueError(f"unknown model variant: {name}")
    if name == "persistence":
        return Persistence()
    if name == "mlp":
        return ResidualMLP()
    if name == "mlp_matched":
        return ResidualMLP(width=160)
    if name == "ssm":
        return StandaloneSSM(horizon)
    return MultiRateNestedWorldModel(
        ModelConfig(
            horizon=horizon,
            share_weights=name not in {"unshared", "detached"},
            modulation=name != "no_modulation",
            detach_bridge=name == "detached",
        )
    )


def nested_loss(model, batch):
    state, summary, predictions, semantic, alignment = batch.initial, None, [], [], []
    flow_losses = []
    for index, actions in enumerate(batch.actions.split(model.config.horizon, 1)):
        output = model(batch.intents, state, actions, batch.context, summary)
        predictions.append(output["pred_states"])
        state, summary = predictions[-1][:, -1], output["sync_summary"]
        semantic.append(
            -(batch.weights * output["objective_weights"].clamp_min(1e-8).log()).sum(-1).mean()
        )
        alignment.append(output["alignment_loss"])
        if model.flow is not None:
            start = index * model.config.horizon
            flow_losses.append(
                model.flow.loss(
                    output["physical_context"],
                    actions,
                    batch.targets[:, start : start + model.config.horizon],
                )
            )
    physics = F.mse_loss(torch.cat(predictions, 1), batch.targets)
    loss = physics + 0.2 * torch.stack(semantic).mean() + 0.01 * torch.stack(alignment).mean()
    if flow_losses:
        loss = loss + 0.05 * torch.stack(flow_losses).mean()
    return loss


@torch.inference_mode()
def evaluate(model, batch):
    prediction = model.rollout(batch.intents, batch.initial, batch.actions, batch.context)
    metrics = {
        "nmse_db": nmse_db(prediction, batch.targets),
        "mse": F.mse_loss(prediction, batch.targets).item(),
        "horizon_mse": (prediction - batch.targets).square().mean((0, 2)).tolist(),
        "state_nmse_db": [nmse_db(prediction[..., i], batch.targets[..., i]) for i in range(6)],
    }
    if isinstance(model, MultiRateNestedWorldModel):
        packet = model.forward_slow_tier(batch.intents, batch.initial, batch.context)
        metrics.update(intent_metrics(packet["objective_weights"], batch.weights))
        metrics.update(planning_audit(model, batch))
    return metrics


@torch.inference_mode()
def counterfactual_audit(model, batch):
    alternate = 1 - batch.actions
    original_prediction = model.rollout(batch.intents, batch.initial, batch.actions, batch.context)
    alternate_prediction = model.rollout(batch.intents, batch.initial, alternate, batch.context)
    state_a, state_b, targets_a, targets_b = batch.initial, batch.initial, [], []
    for index in range(batch.actions.shape[1]):
        state_a = transition(state_a, batch.actions[:, index], batch.context)
        state_b = transition(state_b, alternate[:, index], batch.context)
        targets_a.append(state_a)
        targets_b.append(state_b)
    truth = torch.stack(targets_b, 1) - torch.stack(targets_a, 1)
    effect = alternate_prediction - original_prediction
    return {
        "action_effect_mse": F.mse_loss(effect, truth).item(),
        "action_effect_nmse_db": nmse_db(effect, truth),
    }


def verify_run(row, arguments, directory, *, device="cpu", require_digest=False, tolerance=1e-5):
    """Re-evaluate a completed run before reusing its metrics or publishing an audit."""
    model = build_model(row["variant"], arguments["horizon"]).to(device)
    load_verified_checkpoint(model, row, directory, require_digest=require_digest)
    model.eval()
    count = 0
    for scenario, split in (("iid", "test"), ("ood", "test"), ("iid", "composition")):
        batch = generate_batch(
            arguments["eval_size"], 2 * arguments["horizon"], 900_000, scenario, split
        ).to(device)
        actual = evaluate(model, batch)
        label = "composition" if split == "composition" else scenario
        if label == "iid":
            actual.update(counterfactual_audit(model, batch))
        count += compare_metrics(
            actual, row[label], tolerance, f"{row['variant']}[{row['seed']}].{label}"
        )
    return count


def run_variant(name, seed, args, device):
    torch.manual_seed(seed)
    model = build_model(name, args.horizon).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    start = time.perf_counter()
    trace = []
    if parameters:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
        model.train()
        for update in range(args.steps):
            # Identical training batches for every model and initialization seed.
            batch = generate_batch(args.batch_size, 2 * args.horizon, 100_000 + update).to(device)
            optimizer.zero_grad(set_to_none=True)
            if isinstance(model, MultiRateNestedWorldModel):
                loss = nested_loss(model, batch)
            else:
                loss = F.mse_loss(
                    model.rollout(batch.intents, batch.initial, batch.actions, batch.context),
                    batch.targets,
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss: {name}, seed={seed}, step={update}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            if update == 0 or (update + 1) % 50 == 0 or update + 1 == args.steps:
                trace.append({"step": update + 1, "loss": loss.item()})
                print(
                    f"{name:14s} seed={seed} step={update + 1}/{args.steps} loss={loss.item():.5f}",
                    flush=True,
                )
    training_seconds = time.perf_counter() - start
    model.eval()
    result = {
        "variant": name,
        "seed": seed,
        "parameters": parameters,
        "training_seconds": training_seconds,
        "training_trace": trace,
    }
    for scenario, intent_split in (("iid", "test"), ("ood", "test"), ("iid", "composition")):
        batch = generate_batch(
            args.eval_size, 2 * args.horizon, 900_000, scenario, intent_split
        ).to(device)
        label = "composition" if intent_split == "composition" else scenario
        result[label] = evaluate(model, batch)
        if label == "iid":
            result[label].update(counterfactual_audit(model, batch))
    if isinstance(model, MultiRateNestedWorldModel):
        single = generate_batch(1, args.horizon, 920_000, intent_split="test").to(device)
        with torch.inference_mode():
            packet = model.forward_slow_tier(single.intents, single.initial, single.context)
        result["latency"] = {
            "slow": latency(
                lambda: model.forward_slow_tier(single.intents, single.initial, single.context),
                device,
                repeats=args.latency_repeats,
            ),
            # Cached packet, batch one, an already-allocated recurrent state.
            "fast": latency(
                lambda: model.forward_fast_tier_step(
                    single.initial,
                    single.actions[:, 0],
                    packet,
                    single.initial.new_zeros(1, model.config.memory),
                ),
                device,
                repeats=args.latency_repeats,
            ),
        }
    if parameters and args.checkpoints:
        checkpoint = Path(args.output) / "checkpoints"
        checkpoint.mkdir(parents=True, exist_ok=True)
        path = checkpoint / f"{name}_{seed}.pt"
        save_checkpoint(
            path,
            {
                "state_dict": model.state_dict(),
                "variant": name,
                "seed": seed,
                "config": asdict(model.config) if hasattr(model, "config") else None,
            },
        )
        result["checkpoint_sha256"] = sha256_file(path)
    print(
        f"DONE {name} seed={seed}: IID={result['iid']['nmse_db']:.2f} dB, "
        f"OOD={result['ood']['nmse_db']:.2f} dB ({training_seconds:.1f}s)",
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--eval-size", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--latency-repeats", type=int, default=100)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or npu (requires torch_npu)")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", default="runs/reference")
    parser.add_argument("--checkpoints", action="store_true")
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument(
        "--resume",
        action="store_true",
        help="reuse verified completed runs; restart unfinished models from their seeds",
    )
    restart.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace the output run's experiment record",
    )
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_size, args.horizon, args.threads) < 1:
        parser.error("steps, batch size, eval size, horizon and threads must be positive")
    if args.latency_repeats < 2:
        parser.error("latency-repeats must be at least 2")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        parser.error("seeds and variants must be unique")
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "protocol": "synthetic-two-link-v1",
        "arguments": vars(args),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "device": str(device),
            "threads": torch.get_num_threads(),
        },
        "source_sha256": source_hashes(),
        "command": sys.argv,
        "results": [],
    }
    try:
        with FileLock(output / ".run.lock", timeout=0):
            path = output / "metrics.json"
            if args.resume:
                if not path.is_file():
                    parser.error("--resume requires an existing metrics.json")
                manifest = load_resume_manifest(path, manifest)
                # Check every completed checkpoint before reusing any row or writing new results.
                for row in manifest["results"]:
                    print(f"CHECK {row['variant']} seed={row['seed']}: completed run", flush=True)
                    verify_run(
                        row, manifest["arguments"], output, device=device, require_digest=True
                    )
                manifest.setdefault("resume_commands", []).append(sys.argv)
            elif not args.overwrite and any(item.name != ".run.lock" for item in output.iterdir()):
                parser.error("output run already contains artifacts; use --resume or --overwrite")
            save_json(path, manifest)
            completed = {(row["variant"], row["seed"]) for row in manifest["results"]}
            for variant in args.variants:
                for seed in args.seeds:
                    if (variant, seed) in completed:
                        print(f"REUSE {variant} seed={seed}: verified completed run", flush=True)
                        continue
                    result = run_variant(variant, seed, args, device)
                    manifest["results"].append(result)
                    save_json(path, manifest)
    except Timeout:
        parser.error(f"another experiment writer holds the lock for {output}")


if __name__ == "__main__":
    main()
