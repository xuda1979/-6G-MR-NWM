"""Re-evaluate every saved final checkpoint and verify computational source hashes."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from mr_nwm.data import generate_batch
from mr_nwm.experiment import build_model, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    manifest = json.loads((args.results / "metrics.json").read_text())
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(manifest["environment"]["threads"])
    torch.use_deterministic_algorithms(True)
    for relative, expected in manifest["source_sha256"].items():
        if relative.startswith("src/") or relative == "train_and_eval.py":
            actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"Computational source changed: {relative}")
    for row in manifest["results"]:
        model = build_model(row["variant"], manifest["arguments"]["horizon"])
        if row["parameters"]:
            path = args.results / "checkpoints" / f"{row['variant']}_{row['seed']}.pt"
            saved = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(saved["state_dict"])
        model.eval()
        for scenario, split in (("iid", "test"), ("ood", "test"), ("iid", "composition")):
            batch = generate_batch(
                manifest["arguments"]["eval_size"],
                2 * manifest["arguments"]["horizon"],
                900_000,
                scenario,
                split,
            )
            actual = evaluate(model, batch)
            label = "composition" if split == "composition" else scenario
            for metric in ("nmse_db", "mse", "intent_tv", "planning_regret"):
                if metric in actual and abs(actual[metric] - row[label][metric]) > args.tolerance:
                    raise RuntimeError(f"Metric mismatch: {row['variant']} {label} {metric}")
        print(f"Verified {row['variant']} seed={row['seed']}")
    print(f"All {len(manifest['results'])} checkpoint/baseline evaluations reproduced.")


if __name__ == "__main__":
    main()
