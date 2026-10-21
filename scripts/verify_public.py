"""Authenticate public-data evidence and replay selected model checkpoints on raw holdouts."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mr_nwm.public_data import load_public_dataset
from mr_nwm.public_forecast import (
    ForecastScaler,
    build_forecaster,
    contract_hash,
    frozen_backbone_hash,
    predict_numpy,
    restore_adapters,
    ridge_forecast,
    select_indices,
)
from mr_nwm.run_store import save_json

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independently_score(prediction, truth):
    if (
        prediction.shape != truth.shape
        or not np.isfinite(prediction).all()
        or not np.isfinite(truth).all()
    ):
        raise ValueError("prediction shape or finite-value failure")
    error = prediction.astype(np.float64) - truth
    squared = error * error
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(squared))),
        "nmse_db": float(
            10
            * np.log10(max(float(squared.sum()) / max(float((truth * truth).sum()), 1e-12), 1e-12))
        ),
        "horizon_mae": np.mean(np.abs(error), axis=0).tolist(),
    }


def verify(directory, cache, *, replay=True):
    manifest = json.loads((directory / "metrics.json").read_text())
    if manifest.get("complete") is not True:
        raise ValueError("incomplete public-data experiment")
    settings = manifest["settings"]
    data = load_public_dataset(
        settings["dataset"],
        cache,
        history=settings["history"],
        horizon=settings["horizon"],
        stride=settings["stride"],
    )
    if data.manifest != manifest["data"]:
        raise ValueError("data manifest mismatch")
    for path, expected in manifest["source_sha256"].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"computational source mismatch: {path}")
    expected_identity = contract_hash(
        {
            "settings": settings,
            "data": data.manifest,
            "sources": manifest["source_sha256"],
            "environment": manifest["environment"],
        }
    )
    if expected_identity != manifest["identity_sha256"]:
        raise ValueError("experiment identity mismatch")
    scaler = ForecastScaler.fit(data.train)
    if scaler.to_dict() != manifest["scaler"]:
        raise ValueError("train-only scaler mismatch")
    indices = select_indices(data.test.sample_ids, settings["test_limit"])
    ids = np.asarray([data.test.sample_ids[i] for i in indices])
    if ids.tolist() != manifest["evaluation"]["test_sample_ids"]:
        raise ValueError("test membership mismatch")
    expected_runs = {(name, None) for name in ["persistence", "ridge"]} | {
        (name, seed) for name in settings["models"] for seed in settings["seeds"]
    }
    actual_runs = [(row["model"], row["seed"]) for row in manifest["runs"]]
    if set(actual_runs) != expected_runs or len(actual_runs) != len(expected_runs):
        raise ValueError("missing or duplicated run")
    encoded = scaler.transform(data.test)
    outcomes = []
    for row in manifest["runs"]:
        if row["model"] in {"numeric", "foundation"} and "checkpoint" not in row:
            raise ValueError("learned forecast row requires a checkpoint")
        artifact = directory / row["predictions"]["path"]
        if digest(artifact) != row["predictions"]["sha256"]:
            raise ValueError("prediction checksum mismatch")
        with np.load(artifact, allow_pickle=False) as saved:
            prediction, truth = saved["prediction"], saved["target"]
            np.testing.assert_array_equal(saved["sample_ids"], ids)
            np.testing.assert_array_equal(truth, data.test.targets[indices])
        score = independently_score(prediction, truth)
        if set(score) != set(row["test"]):
            raise ValueError("metric schema mismatch")
        for key, value in score.items():
            np.testing.assert_allclose(value, row["test"][key], rtol=0, atol=1e-10)
        if "checkpoint" in row:
            checkpoint = directory / row["checkpoint"]["path"]
            if digest(checkpoint) != row["checkpoint"]["sha256"]:
                raise ValueError("checkpoint checksum mismatch")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            for key in ["identity_sha256", "model", "seed"]:
                expected = manifest[key] if key == "identity_sha256" else row[key]
                if saved[key] != expected:
                    raise ValueError(f"checkpoint metadata mismatch: {key}")
            if saved["scaler"] != manifest["scaler"]:
                raise ValueError("checkpoint scaler mismatch")
            if saved["selected_step"] != row["training"]["selected_step"]:
                raise ValueError("selected-checkpoint step mismatch")
            if saved["nominal_period_seconds"] != row["training"]["nominal_period_seconds"]:
                raise ValueError("checkpoint cadence mismatch")
            if replay:
                model = build_forecaster(
                    row["model"], history=settings["history"], horizon=settings["horizon"]
                )
                restore_adapters(model, saved["adapters"])
                if frozen_backbone_hash(model) != row["training"]["frozen_backbone_sha256"]:
                    raise ValueError("pretrained backbone fingerprint mismatch")
                actual = scaler.inverse(
                    predict_numpy(model, encoded, indices, saved["nominal_period_seconds"])
                )
                np.testing.assert_allclose(actual, prediction, rtol=1e-5, atol=1e-5)
                del model
        elif replay and row["model"] == "persistence":
            actual = np.repeat(data.test.history[indices, -1, :1], settings["horizon"], 1)
            np.testing.assert_array_equal(actual, prediction)
        elif replay:
            actual, _ = ridge_forecast(scaler.transform(data.train), encoded)
            np.testing.assert_allclose(
                scaler.inverse(actual[indices]), prediction, rtol=1e-8, atol=1e-8
            )
        outcomes.append(
            {
                "model": row["model"],
                "seed": row["seed"],
                "result": "PASS",
                "prediction_values": int(prediction.size),
                "replayed": replay,
            }
        )
    return {
        "dataset": settings["dataset"],
        "result": "PASS",
        "runs": outcomes,
        "identity_sha256": manifest["identity_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/public_data"))
    parser.add_argument("--cache", type=Path, default=Path("runs/public-data-cache"))
    parser.add_argument(
        "--no-replay",
        action="store_true",
        help="authenticate and recompute saved metrics, without executing checkpoints",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/public-verification.json"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    rows = []
    for name in ["ucc4g", "ucc5g", "cicv5g"]:
        rows.append(verify(args.results / name, args.cache, replay=not args.no_replay))
        print(
            json.dumps({"dataset": name, "result": "PASS", "replay": not args.no_replay}),
            flush=True,
        )
    save_json(args.output, {"result": "PASS", "replay": not args.no_replay, "datasets": rows})


if __name__ == "__main__":
    main()
