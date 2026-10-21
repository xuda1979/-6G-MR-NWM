"""Render measured-data tables and figures from authenticated experiment records."""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_public import verify


def summarize(directory, cache=Path("runs/public-data-cache")):
    datasets = {}
    shared_settings, shared_sources = None, None
    for name in ["ucc4g", "ucc5g", "cicv5g"]:
        record = json.loads((directory / name / "metrics.json").read_text())
        if record.get("complete") is not True:
            raise ValueError(f"incomplete experiment: {name}")
        settings = record["settings"]
        common = {key: value for key, value in settings.items() if key != "dataset"}
        if settings["seeds"] != [11, 22, 33] or settings["models"] != ["numeric", "foundation"]:
            raise ValueError("report requires the recorded three-seed, two-neural-model contract")
        if shared_settings is not None and common != shared_settings:
            raise ValueError("inconsistent cross-source experiment settings")
        if shared_sources is not None and record["source_sha256"] != shared_sources:
            raise ValueError("inconsistent cross-source computational sources")
        shared_settings, shared_sources = common, record["source_sha256"]
        verify(directory / name, cache, replay=False)
        scores = {}
        for model in ["persistence", "ridge", "numeric", "foundation"]:
            rows = [row for row in record["runs"] if row["model"] == model]
            expected = 1 if model in {"persistence", "ridge"} else 3
            if len(rows) != expected:
                raise ValueError(f"missing/duplicated runs: {name}:{model}")
            metrics = {}
            for metric in ["mae", "rmse", "nmse_db"]:
                values = [row["test"][metric] for row in rows]
                metrics[metric] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else None,
                }
            metrics["parameters"] = rows[0].get("parameters", 0)
            scores[model] = metrics
        datasets[name] = {
            "unit": record["target_unit"],
            "test_windows": record["evaluation"]["test_windows"],
            "train_windows": record["data"]["splits"]["train"]["windows"],
            "identity_sha256": record["identity_sha256"],
            "models": scores,
        }
    return datasets


def formatted(value, digits=3):
    result = f"{value['mean']:.{digits}f}"
    if value["std"] is not None:
        result += f" ± {value['std']:.{digits}f}"
    return result


def report(directory, paper, cache=Path("runs/public-data-cache")):
    summary = summarize(directory, cache)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    labels = {
        "persistence": "Persistence",
        "ridge": "Ridge",
        "numeric": "Numeric MR-NWM",
        "foundation": "Frozen-LM MR-NWM",
    }
    lines = [
        "# Public measured-data forecasting results",
        "",
        "Five future observations from eight past observations. Lower MAE is better.",
        "Learned-model values are mean ± sample SD over seeds 11, 22, 33; deterministic",
        "controls have one result. These are observational forecasts, not causal RAN control.",
        "",
        "| Model | UCC4G MAE (Mbps) | UCC5G MAE (Mbps) | CICV5G MAE (ms) |",
        "|---|---:|---:|---:|",
    ]
    tex = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & UCC4G (Mbps) & UCC5G (Mbps) & CICV5G (ms)\\",
        r"\midrule",
    ]
    for model, label in labels.items():
        numbers = [formatted(summary[name]["models"][model]["mae"]) for name in summary]
        lines.append(f"| {label} | " + " | ".join(numbers) + " |")
        tex.append(label + " & " + " & ".join(x.replace("±", r"$\pm$") for x in numbers) + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    lines += [
        "",
        "## Evaluation and evidence",
        "",
        "| Dataset | Eligible training windows | Evaluated test windows |",
        "|---|---:|---:|",
    ]
    for name, data in summary.items():
        lines.append(f"| {name} | {data['train_windows']:,} | {data['test_windows']:,} |")
    lines += [
        "",
        "Every learned run uses 300 updates, batch 8, and a checkpoint selected by",
        "validation MAE. All eligible training windows are sampled uniformly with",
        "replacement; actual unique-window coverage is recorded per run. The test cap",
        "is 512 stable-ID windows; all 229 available UCC4G windows are evaluated.",
        "",
        "The language trunk is SmolLM2-135M at the recorded immutable revision and stays",
        "frozen. Numeric adapters, modulation and fast dynamics are trained. The intent",
        "head has no target labels here. This comparison does not isolate pretraining's",
        "causal benefit and does not establish general language understanding.",
        "",
        "Raw predictions, measured targets, sample IDs, source/checkpoint hashes and",
        "per-seed metrics are in each dataset subdirectory. Their upstream data terms",
        "still apply. See [protocol and sources](../../docs/PUBLIC_DATA.md).",
        "",
        "## Interpretation",
        "",
    ]
    for name, data in summary.items():
        scores = data["models"]
        best = min(scores, key=lambda model: scores[model]["mae"]["mean"])
        baseline = scores["persistence"]["mae"]["mean"]
        effect = 100 * (baseline - scores["foundation"]["mae"]["mean"]) / baseline
        lines.append(
            f"- {name}: lowest mean MAE is {labels[best]}; Frozen-LM MR-NWM changes "
            f"MAE by {effect:+.1f}% relative to persistence "
            "(positive denotes lower error; no statistical superiority claim)."
        )
    lines += [
        "",
        "The cohort excludes missing targets and large timestamp gaps. Seed SD does",
        "not quantify site/session population uncertainty. The archived synthetic study",
        "retains its original results and cannot be numerically pooled with this study.",
        "",
    ]
    (directory / "SUMMARY.md").write_text("\n".join(lines))
    root = paper.parent
    chinese_labels = {
        "persistence": "持久预测",
        "ridge": "岭回归",
        "numeric": "数值 MR-NWM",
        "foundation": "冻结语言骨干 MR-NWM",
    }
    chinese = [
        "| 模型 | UCC4G MAE (Mbps) | UCC5G MAE (Mbps) | CICV5G MAE (ms) |",
        "|---|---:|---:|---:|",
    ]
    english = [
        "| Model | UCC4G MAE (Mbps) | UCC5G MAE (Mbps) | CICV5G MAE (ms) |",
        "|---|---:|---:|---:|",
    ]
    for model in labels:
        cells = " | ".join(formatted(summary[name]["models"][model]["mae"]) for name in summary)
        chinese.append(f"| {chinese_labels[model]} | {cells} |")
        english.append(f"| {labels[model]} | {cells} |")
    for path, table in [(root / "README.md", english), (root / "docs/OVERVIEW_CN.md", chinese)]:
        source = path.read_text()
        block = (
            "<!-- BEGIN_PUBLIC_RESULTS -->\n" + "\n".join(table) + "\n<!-- END_PUBLIC_RESULTS -->"
        )
        pattern = r"<!-- BEGIN_PUBLIC_RESULTS -->.*?<!-- END_PUBLIC_RESULTS -->"
        if re.search(pattern, source, flags=re.S):
            source = re.sub(pattern, lambda _: block, source, flags=re.S)
        elif path.name == "README.md":
            source = source.replace("## What is implemented", block + "\n\n## What is implemented")
        else:
            raise ValueError(f"missing generated-result block: {path}")
        path.write_text(source)
    paper.mkdir(parents=True, exist_ok=True)
    (paper / "public_results.tex").write_text("\n".join(tex) + "\n")
    discussion = [
        r"Table~\ref{tab:public} and Figure~\ref{fig:public} report the frozen",
        "test cohort. The three sources have different target units and are not pooled.",
    ]
    for name, data in summary.items():
        scores = data["models"]
        lm = scores["foundation"]["mae"]
        numeric = scores["numeric"]["mae"]
        persist = scores["persistence"]["mae"]["mean"]
        best = min(scores, key=lambda model: scores[model]["mae"]["mean"])
        direction = "lower" if lm["mean"] < persist else "higher"
        percent = abs(100 * (lm["mean"] - persist) / persist)
        discussion.append(
            f"For {name.upper()}, the language-backed mean MAE is "
            f"${lm['mean']:.3f}\\pm{lm['std']:.3f}$~{data['unit']}, "
            f"{percent:.1f}\\% {direction} than persistence. "
            f"The numeric model reaches ${numeric['mean']:.3f}\\pm{numeric['std']:.3f}$. "
            f"{labels[best]} has the lowest mean error among these controls."
        )
    discussion += [
        "These are descriptive comparisons on a fixed, gap-filtered cohort.",
        "Three initialization seeds do not establish a population-level superiority",
        "claim, and a larger frozen backbone is not uniformly justified by this study.",
        "Its additional computational cost must be measured against the value of",
        "a particular task and the strong simple baselines.",
    ]
    (paper / "public_discussion.tex").write_text("\n".join(discussion) + "\n")
    figures = paper / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(9.7, 2.85), constrained_layout=True)
    colors = ["#b3bac3", "#6688a6", "#3768a0", "#188b88"]
    for ax, (name, data) in zip(axes, summary.items()):
        means = [data["models"][model]["mae"]["mean"] for model in labels]
        stds = [data["models"][model]["mae"]["std"] or 0 for model in labels]
        ax.bar(range(4), means, yerr=stds, capsize=3, color=colors)
        ax.set_xticks(range(4), ["Persist", "Ridge", "Numeric", "Frozen LM"], fontsize=8)
        ax.set_title(name.upper(), fontsize=10)
        ax.set_ylabel(f"MAE ({data['unit']})", fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.savefig(figures / "public_forecast.pdf")
    fig.savefig(figures / "public_forecast.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/public_data"))
    parser.add_argument("--paper", type=Path, default=Path("paper"))
    parser.add_argument("--cache", type=Path, default=Path("runs/public-data-cache"))
    args = parser.parse_args()
    report(args.results, args.paper, args.cache)
