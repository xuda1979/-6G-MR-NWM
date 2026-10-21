"""Train measured-trace forecasts; publish raw predictions and authenticated adapters."""

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mr_nwm.public_data import load_public_dataset
from mr_nwm.public_forecast import (
    LANGUAGE_MODEL,
    LANGUAGE_REVISION,
    ForecastScaler,
    adapter_state,
    build_forecaster,
    contract_hash,
    physical_metrics,
    predict_numpy,
    ridge_forecast,
    select_indices,
    train_forecaster,
)
from mr_nwm.run_store import save_checkpoint, save_json

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    "scripts/train_public.py",
    *[
        f"src/mr_nwm/{name}.py"
        for name in (
            "public_data",
            "public_sources",
            "public_forecast",
            "model",
            "foundation",
            "modalities",
            "experts",
            "flow",
            "run_store",
            "verification",
        )
    ],
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_predictions(directory, name, prediction, truth, split, indices):
    path = directory / f"{name}_predictions.npz"
    np.savez_compressed(
        path,
        prediction=prediction.astype(np.float64),
        target=truth.astype(np.float64),
        sample_ids=np.asarray([split.sample_ids[i] for i in indices]),
        group_ids=np.asarray([split.group_ids[i] for i in indices]),
    )
    return {"path": path.name, "sha256": sha256(path)}


def run(args):
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    data = load_public_dataset(args.dataset, args.cache, history=8, horizon=5, stride=5)
    scaler = ForecastScaler.fit(data.train)
    train, validation, test = [
        scaler.transform(split) for split in (data.train, data.validation, data.test)
    ]
    val_ids = select_indices(data.validation.sample_ids, args.validation_limit)
    test_ids = select_indices(data.test.sample_ids, args.test_limit)
    sources = {path: sha256(ROOT / path) for path in SOURCES}
    settings = {
        "protocol": "public-wireless-forecast-v1",
        "dataset": args.dataset,
        "history": 8,
        "horizon": 5,
        "stride": 5,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "learning_rate": 0.001,
        "validation_every": args.validation_every,
        "seeds": args.seeds,
        "models": args.models,
        "validation_limit": args.validation_limit,
        "test_limit": args.test_limit,
        "ridge_alpha": 10.0,
        "language_model": LANGUAGE_MODEL,
        "language_revision": LANGUAGE_REVISION,
        "primary_metric": "physical-unit MAE over all selected windows and 5 horizons",
        "selection": "minimum validation physical-unit MAE; test evaluated after selection",
        "scaler_fit": "observed training histories only; log1p target channel",
        "control_actions": "zero placeholders; no causal radio-control claim",
        "sampling": "uniform training windows with replacement; all train windows eligible",
        "evaluation_sampling": "smallest SHA256(sample_id), immutable across models/seeds",
    }
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "threads": args.threads,
    }
    identity = contract_hash(
        {
            "settings": settings,
            "data": data.manifest,
            "sources": sources,
            "environment": environment,
        }
    )
    manifest = {
        "protocol": settings["protocol"],
        "identity_sha256": identity,
        "settings": settings,
        "source_sha256": sources,
        "environment": environment,
        "data": data.manifest,
        "feature_names": data.feature_names,
        "target_unit": data.target_unit,
        "scaler": scaler.to_dict(),
        "evaluation": {
            "validation_windows": len(val_ids),
            "test_windows": len(test_ids),
            "validation_sample_ids": [data.validation.sample_ids[i] for i in val_ids],
            "test_sample_ids": [data.test.sample_ids[i] for i in test_ids],
        },
        "runs": [],
    }
    manifest_file = args.output / "metrics.json"
    if manifest_file.exists():
        if not args.resume:
            raise ValueError("output exists; use --resume or a new output directory")
        old = json.loads(manifest_file.read_text())
        if old["identity_sha256"] != identity:
            raise ValueError("experiment identity mismatch; use a new output directory")
        manifest["runs"] = old["runs"]
        for row in old["runs"]:
            for field in ("predictions", "checkpoint"):
                if (
                    field in row
                    and sha256(args.output / row[field]["path"]) != row[field]["sha256"]
                ):
                    raise ValueError("completed artifact integrity mismatch")
    save_json(manifest_file, manifest)
    truth = data.test.targets[test_ids].astype(np.float64)

    for name in ("persistence", "ridge"):
        if any(row["model"] == name for row in manifest["runs"]):
            continue
        if name == "persistence":
            prediction = np.repeat(data.test.history[test_ids, -1, :1], 5, axis=1)
        else:
            encoded_prediction, coefficients = ridge_forecast(train, test)
            prediction = scaler.inverse(encoded_prediction[test_ids])
            np.savez_compressed(args.output / "ridge_coefficients.npz", coefficients=coefficients)
        row = {
            "model": name,
            "seed": None,
            "test": physical_metrics(prediction, truth),
            "predictions": save_predictions(
                args.output, name, prediction, truth, data.test, test_ids
            ),
        }
        manifest["runs"].append(row)
        save_json(manifest_file, manifest)
        print(json.dumps({"dataset": args.dataset, "model": name, "test": row["test"]}), flush=True)

    for kind in args.models:
        for seed in args.seeds:
            if any(row["model"] == kind and row["seed"] == seed for row in manifest["runs"]):
                continue
            torch.manual_seed(seed)
            model = build_forecaster(kind)
            config = {**settings, "seed": seed}
            print(
                json.dumps(
                    {
                        "dataset": args.dataset,
                        "model": kind,
                        "seed": seed,
                        "eligible_train_windows": len(train["initial"]),
                    }
                ),
                flush=True,
            )
            training = train_forecaster(
                model,
                train,
                validation,
                val_ids,
                scaler,
                config,
                args.output / "recovery" / f"{kind}_{seed}",
                f"{identity}:{kind}:{seed}",
            )
            prediction = scaler.inverse(
                predict_numpy(model, test, test_ids, training["nominal_period_seconds"])
            )
            checkpoint = args.output / f"{kind}_{seed}_adapters.pt"
            save_checkpoint(
                checkpoint,
                {
                    "identity_sha256": identity,
                    "model": kind,
                    "seed": seed,
                    "config": asdict(model.config),
                    "scaler": scaler.to_dict(),
                    "language_model": LANGUAGE_MODEL if kind == "foundation" else None,
                    "language_revision": LANGUAGE_REVISION if kind == "foundation" else None,
                    "pretrained_identity": getattr(model, "pretrained_identity", None),
                    "adapters": adapter_state(model),
                    "selected_step": training["selected_step"],
                    "nominal_period_seconds": training["nominal_period_seconds"],
                },
            )
            row = {
                "model": kind,
                "seed": seed,
                "training": training,
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "test": physical_metrics(prediction, truth),
                "checkpoint": {"path": checkpoint.name, "sha256": sha256(checkpoint)},
                "predictions": save_predictions(
                    args.output, f"{kind}_{seed}", prediction, truth, data.test, test_ids
                ),
            }
            manifest["runs"].append(row)
            save_json(manifest_file, manifest)
            print(
                json.dumps(
                    {"dataset": args.dataset, "model": kind, "seed": seed, "test": row["test"]}
                ),
                flush=True,
            )
            del model
    save_json(manifest_file, {**manifest, "complete": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["ucc4g", "ucc5g", "cicv5g"])
    parser.add_argument("--cache", type=Path, default=Path("runs/public-data-cache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-every", type=int, default=30)
    parser.add_argument("--validation-limit", type=int, default=256)
    parser.add_argument("--test-limit", type=int, default=512)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument(
        "--models", nargs="+", choices=["numeric", "foundation"], default=["numeric", "foundation"]
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        min(
            args.updates,
            args.batch_size,
            args.validation_every,
            args.validation_limit,
            args.test_limit,
            args.threads,
        )
        <= 0
    ):
        parser.error("budgets and sizes must be positive")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.models)) != len(args.models):
        parser.error("duplicate seeds/models are not allowed")
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output / ".writer.lock"), timeout=0):
        run(args)


if __name__ == "__main__":
    main()
