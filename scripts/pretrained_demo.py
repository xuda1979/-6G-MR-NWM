"""Validate real pretrained LM loading and multimodal/fast-tier wiring, not RAN accuracy.

Numeric adapters and prediction heads are new, untrained parameters. A successful
run establishes API/gradient compatibility only. Model weights are not bundled.
"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch

from mr_nwm.data import generate_batch
from mr_nwm.foundation import LanguageBasedWorldModel
from mr_nwm.modalities import ModalityObservation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("runs/pretrained-integration.json"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(29)
    start = time.perf_counter()
    model = (
        LanguageBasedWorldModel.from_pretrained(
            args.model,
            revision=args.revision,
            modality_dimensions={"csi": 4, "iq": 2, "topology": 3},
        )
        .to(args.device)
        .eval()
    )
    batch = generate_batch(1, 5, seed=120).to(args.device)
    obs = {
        "csi": ModalityObservation(
            torch.randn(1, 3, 4, device=args.device), torch.zeros(1, 3, device=args.device), 0.001
        ),
        "iq": ModalityObservation(
            torch.randn(1, 4, 2, device=args.device), torch.zeros(1, 4, device=args.device), 0.0001
        ),
        "topology": ModalityObservation(
            torch.randn(1, 2, 3, device=args.device),
            torch.zeros(1, 2, device=args.device),
            1.0,
            adjacency=torch.ones(1, 2, 2, device=args.device),
        ),
    }
    inputs = model.tokenize(["Prioritize throughput and energy."], modalities=obs)
    # Nonzero diagnostic head is required to audit causal wiring before task training.
    torch.nn.init.normal_(model.delta.weight, std=0.01)
    output = model(inputs, batch.initial, batch.actions, batch.context)
    loss = (output["pred_states"] - batch.targets).square().mean()
    loss.backward()
    adapter_gradient = model.backbone.modalities.adapters["csi"][1].weight.grad
    assert adapter_gradient is not None and adapter_gradient.abs().sum() > 0
    assert all(p.grad is None for p in model.backbone.language_model.parameters())
    with torch.inference_mode():
        first = model(inputs, batch.initial, batch.actions, batch.context)["pred_states"]
        alternative = model.tokenize(["Prioritize latency."], modalities=obs)
        second = model(alternative, batch.initial, batch.actions, batch.context)["pred_states"]
        assert torch.equal(first, second), "preferences leaked into physical predictions"
    root = Path(__file__).resolve().parents[1]
    sources = [root / "src/mr_nwm/foundation.py", root / "src/mr_nwm/modalities.py", Path(__file__)]
    result = {
        "kind": "pretrained-integration-only",
        "pretrained": model.pretrained_identity,
        "language_parameters": sum(p.numel() for p in model.backbone.language_model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "modalities": list(obs),
        "forecast_shape": list(first.shape),
        "numeric_adapter_gradient_norm": adapter_gradient.norm().item(),
        "preference_invariance": True,
        "frozen_language_weights": True,
        "elapsed_seconds_including_loading": time.perf_counter() - start,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": args.device,
        "source_sha256": {
            str(p.resolve().relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "limitation": (
            "Random numeric adapters and prediction heads; no RAN performance evaluation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
