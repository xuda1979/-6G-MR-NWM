"""Derive manuscript tables and figures from raw metrics, without editing any scores."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

NAMES = {
    "full": "MR-NWM shared",
    "unshared": "MR-NWM unshared",
    "no_modulation": "No modulation",
    "detached": "Detached bridge",
    "mlp": "Residual MLP",
    "mlp_matched": "Matched-size MLP",
    "ssm": "Standalone recurrent",
    "persistence": "Persistence",
}


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def display(pair, digits=2, latex=False):
    separator = r" $\pm$ " if latex else " ± "
    return f"{pair[0]:.{digits}f}{separator}{pair[1]:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    manifest = json.loads(args.metrics.read_text())
    grouped = defaultdict(list)
    for row in manifest["results"]:
        grouped[row["variant"]].append(row)
    table, summary = [], {}
    tex = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & Parameters & IID NMSE (dB) & OOD NMSE (dB) \\",
        r"\midrule",
    ]
    for variant, rows in grouped.items():
        iid = mean_sd([row["iid"]["nmse_db"] for row in rows])
        ood = mean_sd([row["ood"]["nmse_db"] for row in rows])
        parameters = rows[0]["parameters"]
        values = {"parameters": parameters, "iid_nmse_db": iid, "ood_nmse_db": ood}
        table.append(f"| {NAMES[variant]} | {parameters:,} | {display(iid)} | {display(ood)} |")
        tex.append(
            f"{NAMES[variant]} & {parameters:,} & {display(iid, latex=True)} & "
            f"{display(ood, latex=True)} " + r"\\"
        )
        if "intent_tv" in rows[0]["iid"]:
            for split in ("iid", "ood", "composition"):
                for metric in (
                    "intent_tv",
                    "intent_cosine",
                    "planning_regret",
                    "dynamics_only_regret",
                    "power_violation_rate",
                ):
                    values[f"{split}_{metric}"] = mean_sd([row[split][metric] for row in rows])
        values["action_effect_nmse_db"] = mean_sd(
            [row["iid"]["action_effect_nmse_db"] for row in rows]
        )
        if "latency" in rows[0]:
            values["latency"] = {
                tier: {
                    quantile: mean_sd([row["latency"][tier][quantile] for row in rows])
                    for quantile in ("p50_ms", "p95_ms", "p99_ms")
                }
                for tier in ("slow", "fast")
            }
        summary[variant] = values
    tex += [r"\bottomrule", r"\end{tabular}"]
    results_dir, paper = args.root / "results", args.root / "paper"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = [
        "# Reproduced synthetic benchmark",
        "",
        "Mean ± sample standard deviation across initialization seeds. Lower NMSE is better.",
        "",
        "| Model | Parameters | IID NMSE (dB) | OOD NMSE (dB) |",
        "|---|---:|---:|---:|",
        *table,
        "",
        "## Intent and decision audit",
        "",
        "TV is total variation distance; regret is within the finite candidate set.",
        "",
        "| Model | Wording TV | Composition TV | IID regret | Action-effect NMSE (dB) |",
        "|---|---:|---:|---:|---:|",
    ]
    semantic_tex = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & Wording TV & Composition TV & IID regret \\",
        r"\midrule",
    ]
    for variant, values in summary.items():
        if "iid_intent_tv" not in values:
            continue
        tv, composition = values["iid_intent_tv"], values["composition_intent_tv"]
        regret = values["iid_planning_regret"]
        report.append(
            f"| {NAMES[variant]} | {display(tv, 3)} | {display(composition, 3)} | "
            f"{display(regret, 4)} | {display(values['action_effect_nmse_db'])} |"
        )
        semantic_tex.append(
            f"{NAMES[variant]} & {display(tv, 3, True)} & "
            f"{display(composition, 3, True)} & {display(regret, 4, True)} " + r"\\"
        )
    semantic_tex += [r"\bottomrule", r"\end{tabular}"]
    report.extend(
        [
            "",
            "## Host-observed batch-one latency",
            "",
            "Mean of per-run quantiles in milliseconds; device synchronized. "
            "Timing varies by host.",
            "",
            "| Model | Slow p50 | Fast p50 | Fast p95 | Fast p99 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for variant, values in summary.items():
        if "latency" not in values:
            continue
        slow, fast = values["latency"]["slow"], values["latency"]["fast"]
        report.append(
            f"| {NAMES[variant]} | {display(slow['p50_ms'], 3)} | "
            f"{display(fast['p50_ms'], 3)} | {display(fast['p95_ms'], 3)} | "
            f"{display(fast['p99_ms'], 3)} |"
        )
    report += [
        "",
        "## Interpretation",
        "",
        "All scores come from `metrics.json`. This two-link normalized simulator does not "
        "establish performance on channel traces, a 3GPP simulator, a RIC, or a DU. "
        "There is no pretrained language model in the experiment. Hard power-budget "
        "compliance follows from candidate filtering, not from learned safety.",
        "",
        "The detached bridge must be compared with the unshared model to isolate "
        "cross-scale gradient feedback; comparing it only with the shared model confounds "
        "gradient flow and parameter sharing.",
        "",
    ]
    (results_dir / "SUMMARY.md").write_text("\n".join(report))
    (paper / "results.tex").write_text("\n".join(tex) + "\n")
    (paper / "intent_results.tex").write_text("\n".join(semantic_tex) + "\n")
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(7, 2.6), constrained_layout=True)
    for axis, split in zip(axes, ("iid", "ood")):
        for variant, rows in grouped.items():
            curve = np.array([row[split]["horizon_mse"] for row in rows])
            axis.plot(np.arange(1, curve.shape[1] + 1), curve.mean(0), label=NAMES[variant])
        axis.set(
            title="Held-out dynamics" if split == "iid" else "Joint distribution shift",
            xlabel="Forecast step",
            ylabel="Mean squared error",
            yscale="log",
        )
        axis.axvline(manifest["arguments"]["horizon"] + 0.5, color="0.6", linestyle=":")
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    figure.savefig(paper / "figures" / "rollout_error.pdf")
    figure.savefig(paper / "figures" / "rollout_error.png", dpi=200)
    plt.close(figure)
    print("\n".join(report))


if __name__ == "__main__":
    main()
