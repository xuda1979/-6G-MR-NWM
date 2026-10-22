"""Stage A: train KPI forecasters on the 5G-IANA UC6 real measurement campaign.

The UC6 dataset has no control-action logs, so this is an observation-
forecasting task, not an action-conditioned control task: past KPI (throughput,
RTT, RSRQ, RSRP) plus mobility context (position, heading, speed) predict the
next ``--horizon`` seconds of KPIs. Reported metrics are MAE/RMSE/NMSE against
a persistence baseline on physically later, whole-session-isolated test
windows. Normalization statistics are fit on training windows only.
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mr_nwm.uc6 import (  # noqa: E402
    EXPECTED_MD5,
    KPI_NAMES,
    _clean_position,
    build_windows,
    load_uc6_csv,
    segment_uc6,
)


def md5_of(path):
    digest = hashlib.md5()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class KPIForecaster(nn.Module):
    """LSTM over normalized history plus context, decoding normalized future KPIs.

    LSTM is preferred over GRU because the Ascend 910B DynamicGRUV2 operator
    fails to build on the installed CANN runtime, while DynamicRNNGrad-backed
    LSTM is exercised by the platform's own training stack.
    """

    def __init__(self, kpi_dim=4, context_dim=5, hidden=64, horizon=4, layers=1):
        super().__init__()
        self.horizon = horizon
        self.encoder = nn.LSTM(kpi_dim, hidden, layers, batch_first=True)
        self.context_gate = nn.Linear(context_dim, hidden)
        self.decoder = nn.Linear(hidden, kpi_dim * horizon)

    def forward(self, history, context):
        encoded, _ = self.encoder(history)
        state = F.silu(encoded[:, -1] + self.context_gate(context))
        prediction = self.decoder(state).reshape(history.shape[0], self.horizon, -1)
        return prediction


def persistence_future(history, horizon):
    """Repeat the last observed normalized KPI value for every future step."""
    return history[:, -1:].expand(-1, horizon, -1)


def metrics_per_kpi(prediction, target, mask, stats, device):
    """MAE/RMSE/NMSE in physical units, using only observed target positions."""
    mean = torch.tensor(stats["kpi_mean"], device=device)
    std = torch.tensor(stats["kpi_std"], device=device)
    pred_raw = prediction * std + mean
    target_raw = target * std + mean
    results = {}
    for index, name in enumerate(KPI_NAMES):
        observed = mask[..., index]
        if observed.sum() == 0:
            continue
        error = pred_raw[..., index][observed] - target_raw[..., index][observed]
        results[name] = {
            "mae": error.abs().mean().item(),
            "rmse": error.square().mean().sqrt().item(),
            "nmse": (
                (error.square().sum() / target_raw[..., index][observed].square().sum()).item()
            ),
        }
    return results


def run(args):
    device = torch.device(args.device)
    if device.type == "npu":
        torch.npu.set_device(0)
    source_md5 = md5_of(args.data)
    if source_md5 != EXPECTED_MD5:
        sys.exit(f"UC6 MD5 mismatch: {source_md5} != {EXPECTED_MD5}")
    table = load_uc6_csv(args.data)
    segments = segment_uc6(table)
    cleaned = _clean_position(table, segments)
    windows = build_windows(cleaned, segments, history=args.history, horizon=args.horizon)
    history, future, context = windows.normalized()
    history, future, context = (
        history.to(device),
        future.to(device),
        context.to(device),
    )
    future_mask = windows.future_mask.to(device)
    splits = windows.splits.to(device)
    train_rows = splits == 0
    test_rows = splits == 2

    manifest = {
        "protocol": "uc6-stage-a-v1",
        "dataset": {
            "path": str(args.data),
            "md5": source_md5,
            "rows": int(table.shape[0]),
            "segments": len(segments),
            "usable_segments": len(windows.segment_table),
            "window_counts": windows.stats["window_counts"],
        },
        "arguments": {
            "history": args.history,
            "horizon": args.horizon,
            "seeds": args.seeds,
            "steps": args.steps,
            "batch": args.batch,
            "hidden": args.hidden,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "device": args.device,
        },
        "results": [],
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    seed_results = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        model = KPIForecaster(
            kpi_dim=windows.kpi_dim,
            context_dim=windows.context_dim,
            hidden=args.hidden,
            horizon=args.horizon,
        ).to(device)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        indices = torch.nonzero(train_rows).squeeze(1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        start = time.perf_counter()
        for step in range(args.steps):
            choice = indices[torch.randint(len(indices), (args.batch,), generator=generator)]
            prediction = model(history[choice], context[choice])
            observed = future_mask[choice]
            if observed.any():
                residual = (prediction - future[choice]).square()
                loss = (residual * observed).sum() / observed.sum().clamp_min(1)
            else:
                loss = F.mse_loss(prediction, future[choice])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step + 1) % max(1, args.steps // 10) == 0:
                print(
                    f"seed={seed} step={step + 1}/{args.steps} loss={loss.item():.5f}",
                    flush=True,
                )
        training_seconds = time.perf_counter() - start
        model.eval()
        with torch.inference_mode():
            prediction = model(history[test_rows], context[test_rows])
            persistence = persistence_future(history[test_rows], args.horizon)
            target = future[test_rows]
            mask = future_mask[test_rows]
            model_metrics = metrics_per_kpi(prediction, target, mask, windows.stats, device)
            baseline_metrics = metrics_per_kpi(persistence, target, mask, windows.stats, device)
        row = {
            "seed": seed,
            "parameters": parameters,
            "training_seconds": training_seconds,
            "model": model_metrics,
            "persistence": baseline_metrics,
        }
        seed_results.append(row)
        manifest["results"].append(row)
        print(f"seed={seed} done in {training_seconds:.1f}s", flush=True)

    # Aggregate mean ± sd across seeds
    def aggregate(rows, side):
        summary = {}
        for name in KPI_NAMES:
            if name not in rows[0][side]:
                continue
            summary[name] = {
                metric: [
                    float(np.mean([row[side][name][metric] for row in rows])),
                    float(np.std([row[side][name][metric] for row in rows], ddof=1))
                    if len(rows) > 1
                    else 0.0,
                ]
                for metric in ("mae", "rmse", "nmse")
            }
        return summary

    manifest["summary"] = {
        "model": aggregate(seed_results, "model"),
        "persistence": aggregate(seed_results, "persistence"),
    }
    destination = output / "uc6_results.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {destination}")
    print(json.dumps(manifest["summary"], indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/uc6/uc6.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/uc6"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
