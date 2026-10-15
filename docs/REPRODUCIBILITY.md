# Reproduction protocol

## Two distinct validation records

`results/metrics.json` records the small-network synthetic benchmark below.
`results/pretrained_integration.json` records the separately executed Qwen2.5-0.5B
checkpoint integration, with resolved model revision and source hashes. The latter
checks interfaces, numeric-adapter gradients through a frozen pretrained trunk,
and preference invariance; it does not report trained RAN performance.

## Reference experiment

The command, environment, source SHA-256 hashes, losses, metric values, per-step
errors, per-state NMSE, and all timed samples are preserved in `results/metrics.json`.
Final weights are in `results/checkpoints/`. Tables and plots are derived by
`scripts/summarize.py`; no score is manually entered into the tables.

| Setting | Value |
|---|---|
| Protocol | `synthetic-two-link-v1` |
| Initialization seeds | 11, 22, 33 |
| Training seed per update | 100000 + zero-based update |
| Test trajectory seed | 900000 |
| Latency input seed | 920000 |
| Updates / batch size | 300 / 48 |
| Micro steps per window / windows | 5 / 2 |
| Training trajectories | 14,400 trajectories, identical across variants/seeds |
| Evaluation trajectories | 256 per reported split |
| Optimizer | AdamW, learning rate 0.002, weight decay 0.0001 |
| Gradient clipping | Global norm 1 |
| TBPTT | State and memory detached every 3 micro steps within a macro window |
| Loss | Trajectory MSE + 0.2 intent cross entropy + 0.01 routing regularizer |
| Model | Width 32; memory 16; heads 4; generated low-rank width 2 |
| Numeric MLP widths | 64 and 160 (the latter approximately parameter matched) |
| Checkpoint selection | Fixed final update; no test-score selection |
| Optional flow | Disabled in reference results |
| Latency | Batch 1, 10 warm-up, 100 samples, synchronization, cached fast packet |

There are 14,400 training trajectories per model (300 × 48), each ten steps long.
Data are generated on the fly, without downloading an external dataset. The
IID/OOD wording tests share intent labels and initial-state generation, and alter
measured physical context as specified in the paper. The composition split uses
only the held-out three-objective combination. The tokenizer vocabulary comes
from training phrases; held-out wording reorders known words.

The confidence characterization is **mean and sample standard deviation across
initialization seeds**, not a population confidence interval. The benchmark's
single shared test sample is not a substitute for multi-site or trace-level
resampling.

## Running and checking

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,paper]'
make lint test smoke verify
make reproduce
python scripts/summarize.py runs/reference/metrics.json
make paper
```

`make reproduce` writes to ignored `runs/reference/` so the committed evidence
stays intact. To regenerate the reference report, run `make report`.
`make verify` checks the computational source hashes and re-evaluates every
committed checkpoint. It does not assert reproducible wall-clock timing.
Across runtime or hardware changes, small floating-point differences are possible.

The standalone recurrent baseline uses the same fast recurrence and truncation
schedule as MR-NWM. Compare `unshared` with `detached` for the gradient bridge;
compare `full` with `no_modulation` for the modulation ablation. The MLP losses
contain only trajectory regression because they have no intent or router heads.
Reported parameter counts count all allocated trainable parameters, including
the modulator retained but bypassed by `no_modulation`. Equal optimizer updates
do not imply equal FLOPs or wall-clock budgets.

## Accelerator support

The code uses portable PyTorch operations. `--device cuda` enables CUDA and
`--device npu:0` imports `torch_npu` before selecting that device. NPU users must
install a compatible vendor PyTorch/torch_npu/CANN combination themselves.
Only CPU execution is validated in this release. Deterministic algorithm mode
may reject unsupported accelerator operations instead of silently changing
behavior. GPU timing also needs the vendor-recommended deterministic workspace
settings for a reproducible training run.

## Source provenance

The input drafts contained assertions and example numbers rather than an
executed experiment record. This release does not reuse those numbers. Its
metrics, checkpoints, plots and generated tables are derived from the actual
reference execution. External method descriptions are cited to primary papers;
no third-party paper text or code is vendored.
