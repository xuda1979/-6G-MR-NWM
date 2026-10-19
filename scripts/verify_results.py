"""Re-evaluate every saved final checkpoint and verify computational source hashes."""

import argparse
import hashlib
import json
import math
import platform
from dataclasses import asdict
from pathlib import Path

import torch

from mr_nwm.data import generate_batch
from mr_nwm.experiment import build_model, counterfactual_audit, evaluate
from mr_nwm.provenance import source_hashes
from mr_nwm.verification import compare_metrics, validate_result_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--allow-source-changes",
        action="store_true",
        help="cross-revision metric regression, without claiming source-matched reproduction",
    )
    parser.add_argument("--output", type=Path, help="write the audit and current source hashes")
    args = parser.parse_args()
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("tolerance must be nonnegative and finite")
    manifest = json.loads((args.results / "metrics.json").read_text())
    validate_result_rows(manifest)
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(manifest["environment"]["threads"])
    torch.use_deterministic_algorithms(True)
    mismatches = {}
    current_sources = source_hashes(root)
    for relative, expected in manifest["source_sha256"].items():
        if relative.startswith("src/") or relative == "train_and_eval.py":
            actual = current_sources.get(relative)
            if actual != expected:
                mismatches[relative] = {"reference": expected, "current": actual}
    if mismatches and not args.allow_source_changes:
        raise RuntimeError(
            f"Computational source changed: {', '.join(mismatches)}. "
            "Use --allow-source-changes for an explicit cross-revision metric regression."
        )
    mode = (
        "cross-revision-regression" if args.allow_source_changes else "source-matched-reproduction"
    )
    print(f"Mode: {mode}; changed reference sources: {len(mismatches)}", flush=True)
    scalar_count = 0
    for row in manifest["results"]:
        model = build_model(row["variant"], manifest["arguments"]["horizon"])
        if sum(parameter.numel() for parameter in model.parameters()) != row["parameters"]:
            raise RuntimeError(f"Parameter count mismatch: {row['variant']}")
        if row["parameters"]:
            path = args.results / "checkpoints" / f"{row['variant']}_{row['seed']}.pt"
            saved = torch.load(path, map_location="cpu", weights_only=True)
            config = asdict(model.config) if hasattr(model, "config") else None
            if (saved["variant"], saved["seed"], saved["config"]) != (
                row["variant"],
                row["seed"],
                config,
            ):
                raise RuntimeError(f"Checkpoint metadata mismatch: {path}")
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
            if label == "iid":
                actual.update(counterfactual_audit(model, batch))
            scalar_count += compare_metrics(
                actual, row[label], args.tolerance, f"{row['variant']}[{row['seed']}].{label}"
            )
        print(f"Verified {row['variant']} seed={row['seed']}")
    print(f"All {len(manifest['results'])} deterministic evaluations matched reference metrics.")
    if args.output is not None:
        report = {
            "mode": mode,
            "reference_metrics_sha256": hashlib.sha256(
                (args.results / "metrics.json").read_bytes()
            ).hexdigest(),
            "source_mismatches": mismatches,
            "current_source_sha256": current_sources,
            "rows_verified": len(manifest["results"]),
            "metric_values_verified": scalar_count,
            "absolute_tolerance": args.tolerance,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "excluded": ["training replay", "wall-clock latency"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
