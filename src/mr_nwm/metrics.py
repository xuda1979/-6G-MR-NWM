"""Explicit normalization, paired intent metrics, and synchronized latency distributions."""

import time

import numpy as np
import torch


def nmse_db(prediction, target):
    """10 log10(sum error^2 / sum target^2), not variance-normalized MSE."""
    if prediction.shape != target.shape or target.numel() == 0:
        raise ValueError("prediction and target must have identical nonempty shapes")
    numerator = (prediction.double() - target.double()).square().sum()
    denominator = target.double().square().sum()
    if denominator == 0:
        raise ValueError("NMSE is undefined for a zero-energy target")
    return (10 * (numerator.clamp_min(1e-30) / denominator).log10()).item()


def intent_metrics(prediction, target):
    """TV=0 is exact. Cosine is a similarity, never relabeled an alignment rate."""
    return {
        "intent_tv": (0.5 * (prediction - target).abs().sum(-1)).mean().item(),
        "intent_cosine": torch.nn.functional.cosine_similarity(prediction, target).mean().item(),
    }


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu":
        torch.npu.synchronize()


def latency(function, device, warmup=10, repeats=100):
    """Host-observed batch-one latency with device synchronization and warm-up."""
    if warmup < 0 or repeats < 2:
        raise ValueError("need nonnegative warmup and at least two repeats")
    with torch.inference_mode():
        for _ in range(warmup):
            function()
        _synchronize(device)
        samples = []
        for _ in range(repeats):
            _synchronize(device)
            start = time.perf_counter_ns()
            function()
            _synchronize(device)
            samples.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "p99_ms": float(np.percentile(samples, 99)),
        "samples_ms": samples,
    }
