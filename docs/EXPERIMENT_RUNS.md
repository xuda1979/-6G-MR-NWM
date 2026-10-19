# Recovering experiment sweeps

The experiment runner preserves completed variant/seed runs across interruptions.
Every reusable row is tied to a fixed experiment contract and a saved checkpoint.

## Reference sweep

```bash
make reproduce
# After an interruption, from the same source revision and runtime:
make resume
```

`make resume` verifies every completed checkpoint's SHA-256 digest, model metadata,
and recomputed deterministic metrics. It then skips training for those rows and
runs the missing variant/seed combinations. An unfinished model starts again from
its original initialization seed and fixed training batches. There is no recovery
of optimizer steps within an unfinished model.

The same command can resume a fully completed sweep: it checks and reuses all rows
without training another model. Raw latency measurements and training durations
for reused rows remain the original measurements.

## Custom experiment

Start with `--checkpoints` so trained models can be verified on a later invocation:

```bash
python train_and_eval.py --steps 50 --seeds 11 22 --variants full ssm \
  --checkpoints --output runs/custom
python train_and_eval.py --steps 50 --seeds 11 22 --variants full ssm \
  --checkpoints --output runs/custom --resume
```

Keep the model variants, seeds, optimizer-update budget, batch/evaluation sizes,
horizon, device, threads and timing settings identical. Code and recorded runtime
fields must also match. A mismatch stops recovery before modifying the experiment
record. Changed experiments belong in a new output directory.

## Output and failure behavior

- A fresh invocation refuses an output directory that already contains artifacts.
  Use `--resume` to recover the same sweep, or deliberately choose `--overwrite`
  to replace its experiment record and the checkpoints written by that invocation.
  Unreferenced older files are retained; they are not part of the new manifest.
- A run-directory lock permits one writer. The operating system releases the lock
  when the process exits; a leftover `.run.lock` file does not itself indicate an
  active writer. Reliable filesystem locking and atomic rename are required.
- A manifest exists before training begins. Checkpoints are saved before their
  completed rows are published. If interruption leaves an unreferenced checkpoint,
  resume trains that missing row again rather than trusting the uncommitted file.
- Temporary writes are flushed and synchronized before atomic replacement. A failed
  write leaves the preceding complete artifact intact. An abrupt kill can leave a
  `.pending-*` temporary file; it is not used as experiment evidence.
- Non-finite losses or gradient norms stop training before committing that model's
  completed result. Rows from models that finished earlier remain recoverable.
- Duplicate seeds or variants are rejected before creating experiment outputs.

## Evidence formats

Version-2 manifests record checkpoint digests and the commands used to resume.
`scripts/verify_results.py` requires recorded digests for version-2 trained models
and reports how many checkpoint digests were checked. Original version-1 benchmark
records remain verifiable through their metadata and deterministic metrics, but
cannot be resumed using this contract. A digest detects a changed checkpoint; it
does not authenticate the person who produced the manifest.

The recovery path is validated on CPU. Accelerator recovery additionally depends
on deterministic operator and numerical behavior of the selected vendor runtime.

## Validation of this change

On Python 3.12 / PyTorch 2.5.1 CPU, all 83 repository tests and Ruff checks pass.
The interruption test preserves the completed row and checkpoint byte-for-byte,
then produces the same model tensors and deterministic metrics as an uninterrupted
four-run sweep. Tests also cover modified arguments, sources, runtime, metrics,
checkpoint bytes/metadata, writer contention, failed atomic replacements and
non-finite gradients.

The public smoke command trains all eight variants, reuses all completed rows in a
`--resume` invocation, and strictly verifies the resulting checkpoints. The resume
check logs no training updates and preserves the recorded results. All 24 original
reference rows still
match across 1,560 deterministic metrics; the audit is recorded in
`results/review2_checkpoint_verification.json`. Historical weights and benchmark
scores retain their original provenance.
