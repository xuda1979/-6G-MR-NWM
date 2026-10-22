# Reproduction protocol

## Three distinct validation records

`results/metrics.json` records the small-network synthetic benchmark below.
`results/pretrained_integration.json` records the separately executed Qwen2.5-0.5B
checkpoint integration, with resolved model revision and source hashes. The latter
checks interfaces, numeric-adapter gradients through a frozen pretrained trunk,
and preference invariance; it does not report trained RAN performance.
`results/uc6_stage_a/` records the real-measurement UC6 Stage-A run
(protocol `uc6-stage-a-v1`): per-seed result JSON files executed on Ascend 910B
NPU hosts, the aggregate `uc6_stage_a_results.json` derived without manual
transcription, and per-seed environment metadata. It is observation
forecasting only; the campaign has no control-action log.
`results/ucc5g_stage_a/` records the second real-measurement campaign
(UCC 5G production drive tests, protocol `ucc5g-stage-a-v1`) with the same
aggregate-from-per-seed discipline.

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

## UC6 Stage-A (real measurement data)

Download `uc6.csv` from Zenodo record 10037164 (CC BY 4.0) into
`data/uc6/`; the loader verifies MD5
`f8047eb5a2d57a5f789363b0c93757e5` and 26,733 rows on load. Then:

```bash
python scripts/train_uc6.py --steps 2000 --seeds 11 22 33 --device cpu \
  --output runs/uc6/uc6_results.json
```

The committed evidence in `results/uc6_stage_a/` was produced on Ascend 910B
NPU hosts (`--device npu`), one seed per invocation, seeds 11/22/33. Test
metrics are computed on chronologically later, whole-session-isolated
segments; normalization fits training segments only. CPU runs are
equivalent for the protocol; only wall-clock differs (~21 s per seed on the
NPU host).

## UCC 5G Stage-A (second real measurement campaign)

Download `5G-production-dataset.zip` from
github.com/uccmisl/5Gdataset (GPL-3.0; codeload archive mirror works when
raw.githubusercontent is blocked), extract under `data/ucc5g/`; the loader
verifies 83 CSV files and 188,711 total rows on load. Then (run once per
seed to mirror the committed evidence layout):

```bash
python scripts/train_ucc5g.py --steps 2000 --seeds 11 --device cpu \
  --output runs/ucc5g/box_runs/seed11_ucc5g
```

`results/ucc5g_stage_a/` records this run (protocol `ucc5g-stage-a-v1`):
per-seed results aggregated without manual transcription, plus environment
metadata. Splits are by whole session file in chronological order
(70/15/15% of 83 sessions); normalization fits training windows only. The
committed evidence was produced on CPU (~3.5 s per seed). Targets: DL
bitrate, RSRP, RSRQ, SNR; ping is excluded (populated in 21 of 83 files
only). As with UC6, this is observation forecasting; the campaign has no
control-action log.


`make reproduce` writes to ignored `runs/reference/` so the committed evidence
stays intact. To regenerate the reference report, run `make report`.
`make verify` performs an explicit **cross-revision regression** against the original
committed checkpoints. It records source differences and current hashes in
`runs/checkpoint-verification.json`; it does not rewrite the reference manifest.
The comparison covers all deterministic metrics, including per-step/per-state vectors,
intent metrics, dynamics-only regret, feasibility, and paired action effects.
Non-finite values, missing metrics, incomplete run grids and checkpoint metadata
mismatches fail the audit. It does not assert reproducible wall-clock timing or
replay optimizer updates.

`make verify-exact`, or `python scripts/verify_results.py` without flags, additionally
requires the recorded computational source hashes to match. It intentionally fails
on changed computational sources; use the recorded source revision for exact replay.
`make smoke` trains all variants for two updates, saves checkpoints, exercises
completed-sweep recovery, and runs this strict verifier against its fresh manifest.
New source manifests hash project Python
files under `src/`, `scripts/`, `tests/` and the repository root; installed virtual
environments and generated run directories are excluded.
Across runtime or hardware changes, small floating-point differences are possible.

## Interrupted sweeps and checkpoint integrity

`make reproduce` creates a new version-2 manifest before starting any model, then
commits each completed variant/seed row. `make resume` first checks the experiment
arguments, recorded source hashes and runtime contract. It verifies checkpoint
digests and metadata, then recomputes every completed row's deterministic metrics
before reusing the row. An incomplete model is trained again from its original
initialization seed; optimizer steps inside that model are not resumed.

Checkpoint and JSON writes use a temporary file, flush and `fsync`, followed by
atomic replacement. An operating-system-backed file lock prevents concurrent
writers in one output directory. A fresh invocation refuses an occupied output
directory unless `--resume` or `--overwrite` is explicit. Gradient clipping rejects
non-finite norms before the optimizer update.

Version-2 checkpoints include SHA-256 digests in their manifest rows; the verifier
checks these before loading model weights. Historical version-1 checkpoints still
support metadata and metric verification, and the audit reports that they lack
recorded digests. Legacy manifests do not support the new resume contract.
See [EXPERIMENT_RUNS.md](EXPERIMENT_RUNS.md) for commands and recovery boundaries.

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
